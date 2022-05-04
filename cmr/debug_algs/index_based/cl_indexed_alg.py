# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from cmr.debug_algs.cl_utils import get_top_interfered_examples, get_virtual_updated_model
from cmr.debug_algs.index_based.IO_each_index import BartIOIndexManager
from cmr.debug_algs.index_based.biencoder import BiEncoderIndexManager
from cmr.debug_algs.index_based.index_manager import BartIndexManager, RandomMemoryManger
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from cmr.debug_algs.cl_simple_alg import ContinualFinetuning
from tqdm import tqdm
import random
import numpy as np
import torch
import transformers
from cmr.task_manager.eval_metrics import evaluate_func
import copy
import pickle
import os
from cmr.models.mybart import MyBart
from cmr.models import run_bart
from cmr.models.utils import (convert_model_to_single_gpu,
                                           freeze_embeds, trim_batch)
from argparse import Namespace
import more_itertools
import json


class IndexBasedCL(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "tbd"

    def _check_debugger_args(self):
        super()._check_debugger_args()
        required_atts = [
            "replay_size",
            "replay_candidate_size",
            "replay_frequency",
            "memory_store_rate",  # 0, 0.1, 1 etc.
            "upstream_sample_ratio",
            "memory_path",  # to save the memory module from disk
            "use_replay_mix",
            "init_memory_cache_path",
            "index_rank_method"
        ]
        assert all([hasattr(self.debugger_args, att) for att in required_atts])
        # assert self.debugger_args.index_rank_method in ["most_similar", "most_different"]

    def debugger_setup(self, debugger_args):

        super().debugger_setup(debugger_args)
        self.memroy_module = None # if online is used seperately
        self.upstream_memroy_module = None


        
        def setup_bart_index():
            mm = BartIndexManager(self.logger)
            mm.set_up_data_args(self.data_args)
            mm.data_args.predict_batch_size = 4
            mm.load_encoder_model(self.base_model_args)
            return mm

        def setup_bart_io_index():
            mm = BartIOIndexManager(self.logger)
            mm.set_up_data_args(self.data_args)
            mm.data_args.predict_batch_size = 4
            mm.load_encoder_model(self.base_model_args)
            return mm


        def setup_biencoder():
            with open(debugger_args.indexing_args_path) as f:
                train_args_dict = json.load(f)
            mm = BiEncoderIndexManager(self.logger)
            mm.train_args = Namespace(**train_args_dict)
            mm.set_up_data_args(self.data_args)
            mm.data_args.predict_batch_size = 4
            mm.load_encoder_model(
                self.base_model_args,
                mm.train_args.memory_encoder_path,
                mm.train_args.query_encoder_path)
            return mm
            
        # Initializing the BartIndexManager
        self.logger.info(f"indexing_method={debugger_args.indexing_method}")
        self.name = f"index_cl_{debugger_args.indexing_method}"
        if debugger_args.indexing_method == "bart_index":
            self.logger.info("setup_bart_index")
            self.upstream_memroy_module = setup_bart_index()
        elif debugger_args.indexing_method == "bart_io_index":
            self.logger.info("bart_io_index")
            self.upstream_memroy_module = setup_bart_io_index()
        elif debugger_args.indexing_method == "biencoder":
            self.logger.info("biencoder")
            self.upstream_memroy_module = setup_biencoder() 

        assert self.upstream_memroy_module is not None

        if debugger_args.init_memory_cache_path:
            self.upstream_memroy_module.load_memory_from_path(debugger_args.init_memory_cache_path)
        else:
            self.upstream_memroy_module.set_up_initial_memory(
                formatted_examples=self.sampled_upstream_examples)
        

        if self.debugger_args.upstream_sample_ratio < 0: 
            self.logger.info("upstream_sample_ratio < 0 ; self.memroy_module <---> self.upstream_memroy_module")
            self.memroy_module = self.upstream_memroy_module
        else:
            self.logger.info("upstream_sample_ratio > 0 ; two seperate memory module")
            if debugger_args.indexing_method == "bart_io_index":
                self.memroy_module = setup_bart_io_index()
            elif debugger_args.indexing_method == "biencoder":
                self.memroy_module = setup_biencoder()
            elif debugger_args.indexing_method == "bart_index":
                self.memroy_module = setup_bart_index()
        return

    def online_debug(self):
        self.logger.info("Start Online Debugging with Dynamic Error Mode")
        self.logger.info(f"Number of Batches of Data: {self.num_data_batches}")
        self.logger.info(f"Data Batch Size: {self.data_batch_size};")
        self.timecode = 0

        if self.debugger_args.save_ckpt_freq > 0 and self.timecode % self.debugger_args.save_ckpt_freq == 0:
            # save the initial model as the 0-th model.
            self._save_base_model()

        self.past_errors = []
        self.past_submission = []
        last_steps = 0
        self.logger.info("Copying initial model")
        initial_model = copy.deepcopy(self.base_model) # for the use of query

        for data_eval_loader in tqdm(self.data_eval_loaders, desc="Online Debugging (with Index-based replay)"):

            result_dict = {"timecode": self.timecode}   # start with 0
            self.eval_knowledge_retention(result_dict)
            self.eval_knowledge_generalization(result_dict)


            ############### CORE ###############

            # self._replay_based_eval(result_dict)
            formatted_bug_examples = self._get_dynamic_errors(
                data_eval_loader, result_dict, return_raw_bug_examples=True)
            _, bug_eval_loader = self.get_dataloader(self.data_args, formatted_bug_batch=formatted_bug_examples, mode="eval")
            
            examples_to_train = formatted_bug_examples[:]

            # if (self.model_update_steps - last_steps) >= self.debugger_args.replay_frequency \
            if self.timecode % self.debugger_args.replay_frequency == 0 \
                    and self.debugger_args.replay_frequency > 0 and self.debugger_args.replay_size > 0 \
                    and self.timecode > 0:
                # sparse experience replay
                self.logger.info("Triggering Sampling from Memory and starting to replay.")
                self.logger.info(f"Current memroy_module size: {self.memroy_module.get_memory_size()}.")
                if self.upstream_memroy_module:
                    self.logger.info(f"Current upstream_memroy_module size: {self.upstream_memroy_module.get_memory_size()}.")
                
                if self.debugger_args.indexing_method == "biencoder":
                    # self.memroy_module.before_model = initial_model   # if for longer-delta
                    self.upstream_memroy_module.before_model = initial_model
                    self.upstream_memroy_module.after_model = get_virtual_updated_model(self, bug_train_loader)
                elif self.debugger_args.indexing_method == "bart_io_index":
                    # self.upstream_memroy_module.bart_model = initial_model
                    if self.debugger_args.upstream_sample_ratio > 0:    # a seperate online memory module
                        self.memroy_module.bart_model = self.base_model 
                elif self.debugger_args.indexing_method == "bart_index":
                    # self.upstream_memroy_module.bart_model = initial_model
                    if self.debugger_args.upstream_sample_ratio > 0:    # a seperate online memory module
                        self.memroy_module.bart_model = self.base_model
                        
                if self.debugger_args.use_mir:
                    assert self.debugger_args.replay_candidate_size >= self.debugger_args.replay_size 
                    def mir_retrieve(mm, sample_size):
                        effective_cand_size = min(self.debugger_args.replay_candidate_size, mm.get_memory_size())
                        self.logger.info(f"effective_cand_size={effective_cand_size}")
                        each_sample_size = int(effective_cand_size*1.1/sample_size)
                        self.logger.info(f"each_sample_size={each_sample_size}")
                        assert effective_cand_size >= self.debugger_args.replay_size
                        retrieved_examples_candidates = mm.retrieve_from_memory(
                            query_examples=formatted_bug_examples,
                            sample_size=effective_cand_size,
                            rank_method=self.debugger_args.index_rank_method,
                            agg_method="each_topk_then_random",
                            each_sample_size=each_sample_size,
                            each_sim_sample_size=min(each_sample_size*5, mm.get_memory_size()), # only used for the bart-IO
                            ) 
                        if "mir_buffer_ids" not in result_dict:
                            result_dict["mir_buffer_ids"] = []    
                        result_dict["mir_buffer_ids"] += [_id for (_input, _truth, _id) in retrieved_examples_candidates]
                        retrieved_examples = get_top_interfered_examples(self,
                            K=sample_size, candidate_examples=retrieved_examples_candidates, query_data_loader=bug_train_loader)
                        return retrieved_examples

                    if self.debugger_args.upstream_sample_ratio > 0:
                        upstream_sample_budget = int(self.debugger_args.upstream_sample_ratio * self.debugger_args.replay_size)
                        self.logger.info(f"Memory from upstream_memroy_module = {upstream_sample_budget}; ")
                        self.logger.info(f"Memory from memroy_module = {self.debugger_args.replay_size-upstream_sample_budget}; ")
                        retrieved_examples = []
                        if upstream_sample_budget > 0:
                            retrieved_examples += mir_retrieve(mm=self.upstream_memroy_module,
                                sample_size=upstream_sample_budget)
                        retrieved_examples += mir_retrieve(mm=self.memroy_module,
                            sample_size=self.debugger_args.replay_size-upstream_sample_budget) 
                    else:
                        retrieved_examples = mir_retrieve(mm=self.memroy_module, sample_size=self.debugger_args.replay_size)

                else:
                    each_sample_size=5
                    each_sim_sample_size=30
                    retrieved_examples = []
                    upstream_sample_budget = 0
                    if self.debugger_args.upstream_sample_ratio > 0:
                        upstream_sample_budget = int(self.debugger_args.upstream_sample_ratio * self.debugger_args.replay_size)
                        self.logger.info(f"Memory from upstream_memroy_module = {upstream_sample_budget}; ")
                        self.logger.info(f"Memory from memroy_module = {self.debugger_args.replay_size-upstream_sample_budget}; ")
                        retrieved_examples += self.upstream_memroy_module.retrieve_from_memory(
                            query_examples=formatted_bug_examples,
                            sample_size=upstream_sample_budget,
                            agg_method="each_topk_then_random",
                            rank_method=self.debugger_args.index_rank_method,
                            each_sample_size=each_sample_size, each_sim_sample_size=each_sim_sample_size)
                    retrieved_examples += self.memroy_module.retrieve_from_memory(
                        query_examples=formatted_bug_examples,
                        sample_size=self.debugger_args.replay_size-upstream_sample_budget,
                        agg_method="each_topk_then_random",
                        rank_method=self.debugger_args.index_rank_method,
                        each_sample_size=each_sample_size, each_sim_sample_size=each_sample_size*5)
                    # self.logger.info(f"retrieved_examples (index)={retrieved_examples}")
                
                result_dict["retrieved_ids"] = [_id for (_input, _truth, _id) in retrieved_examples]

                if self.debugger_args.use_replay_mix:
                    examples_to_train += retrieved_examples
                    self.logger.info(
                        f"Mixed the retrieved examples (len={len(retrieved_examples)}) to the current batch for training.")
                else:
                    self.logger.info(
                        f"Replay-Training Start! Using the retrieved examples (len={len(retrieved_examples)})  ")
                    replay_data_loader, _ = self.get_dataloader(
                        self.data_args, retrieved_examples, mode="train")
                    self.fix_bugs(replay_data_loader, quiet=False)  # sparse replay
                    self.logger.info("Replay-Training done.")

            last_steps = self.model_update_steps 

            # Fix the bugs by mini-batch based "training"
            self.logger.info(
                f"Start error-fixing (len(examples_to_train)={len(examples_to_train)}) .... Timecode: {self.timecode}")
            bug_train_loader, _ = self.get_dataloader(
                self.data_args, examples_to_train, mode="train")
            self.fix_bugs(bug_train_loader)   # for debugging
            self.logger.info("Start error-fixing .... Done!")
            
            flag_store_examples = True
            if flag_store_examples:
                self.logger.info(
                    f"Saving the current error examples (len={len(formatted_bug_examples)}) to the memory.")
                self.logger.info(f"Current memroy_module size: {self.memroy_module.get_memory_size()}.")
                if self.upstream_memroy_module:
                    self.logger.info(f"Current upstream_memroy_module size: {self.upstream_memroy_module.get_memory_size()}.")
                self.memroy_module.store_examples(formatted_bug_examples)
                self.logger.info("Finished.")

            ############### CORE ###############

            
            self.evaluate_error_fixing(result_dict, bug_eval_loader)
            self._update_result_dict(result_dict)
            
            if self.debugger_args.save_ckpt_freq > 0 and self.timecode % self.debugger_args.save_ckpt_freq == 0:
                self._save_base_model()
                self.save_result_file()
            self.logger.info("-"*50)
            self.timecode += 1


        #### Final evaluation ####
        self.final_evaluation()
        
        #### Save the final model ####
        self._save_base_model()

        # Save to path
        self.memroy_module.save_memory_to_path(self.debugger_args.memory_path)
