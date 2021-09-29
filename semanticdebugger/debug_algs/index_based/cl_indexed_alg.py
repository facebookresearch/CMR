from semanticdebugger.debug_algs.index_based.index_manager import IndexManager
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
from tqdm import tqdm
import random
import numpy as np
import torch
import transformers
from semanticdebugger.task_manager.eval_metrics import evaluate_func
import copy
import pickle
import os
from semanticdebugger.models.mybart import MyBart
from semanticdebugger.models import run_bart
from semanticdebugger.models.utils import (convert_model_to_single_gpu,
                                           freeze_embeds, trim_batch)
from argparse import Namespace
import more_itertools


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
            "memory_path",  # to save the memory module from disk 
            "use_replay_mix",
            "init_memory_cache_path",
            "index_rank_method"
        ]
        assert all([hasattr(self.debugger_args, att) for att in required_atts])

    def debugger_setup(self, debugger_args):

        super().debugger_setup(debugger_args)

        # Initializing the IndexManager
        self.memroy_module = IndexManager(self.logger)
        self.memroy_module.set_up_data_args(self.data_args)
        self.memroy_module.data_args.predict_batch_size = 8
        self.memroy_module.load_encoder_model(self.base_model_args)
        if debugger_args.init_memory_cache_path:
            self.memroy_module.load_memory_from_path(debugger_args.init_memory_cache_path)
        else:
            self.memroy_module.initial_memory_path = self.data_args.sampled_upstream_json_path
            self.memroy_module.set_up_initial_memory()
        return

    def online_debug(self):
        self.logger.info("Start Online Debugging with Dynamic Error Mode")
        self.logger.info(f"Number of Batches of Data: {self.num_data_batches}")
        self.logger.info(f"Data Batch Size: {self.data_batch_size};")
        self.timecode = 0

        if self.debugger_args.save_all_ckpts:
            # save the initial model as the 0-th model.
            self._save_base_model()

        self.overall_errors = []
        self.seen_stream_data = []
        last_steps = 0

        for data_eval_loader in tqdm(self.data_eval_loaders, desc="Online Debugging (Dynamic)"):

            result_dict = {"timecode": self.timecode}   # start with 0

            self._replay_based_eval(result_dict)
            formatted_bug_examples = self._get_dynamic_errors(
                data_eval_loader, result_dict, return_raw_bug_examples=True)

            examples_to_train = formatted_bug_examples[:]

            # if (self.model_update_steps - last_steps) >= self.debugger_args.replay_frequency \
            if self.timecode % self.debugger_args.replay_frequency == 0 \
                    and self.debugger_args.replay_frequency > 0 and self.debugger_args.replay_size > 0 \
                    and self.timecode > 0:
                # sparse experience replay
                self.logger.info("Triggering Sampling from Memory and starting to replay.")
                self.logger.info(f"Current memory size: {len(self.memroy_module.memory_index_sorted_ids)}.")

                retrieved_examples = self.memroy_module.retrieve_from_memory(
                    query_examples=formatted_bug_examples,
                    sample_size=self.debugger_args.replay_size,
                    rank_method=self.debugger_args.index_rank_method)

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

            ############### CORE ###############
            # Fix the bugs by mini-batch based "training"
            self.logger.info(
                f"Start bug-fixing (len(examples_to_train)={len(examples_to_train)}) .... Timecode: {self.timecode}")
            bug_train_loader, _ = self.get_dataloader(
                self.data_args, examples_to_train, mode="train")
            self.fix_bugs(bug_train_loader)   # for debugging
            self.logger.info("Start bug-fixing .... Done!")
            ############### CORE ###############
            self._log_episode_result(result_dict, data_eval_loader)
            self.timecode += 1

            if self.debugger_args.save_all_ckpts:
                self._save_base_model()

            # Store to memory
            _max = 1000000
            flag_store_examples = bool(random.randrange(0, _max)/_max >=
                                       1 - self.debugger_args.memory_store_rate)
            if flag_store_examples:
                self.logger.info(
                    f"Saving the current error examples (len={len(formatted_bug_examples)}) to the memory.")
                self.logger.info(f"Current memory size: {len(self.memroy_module.memory_index_sorted_ids)}.")
                self.memroy_module.store_exampls(formatted_bug_examples)
                self.logger.info("Finished.")
            self.logger.info("-"*50)
        #### Final evaluation ####
        self.final_evaluation()

        # Save to path
        self.memroy_module.save_memory_to_path(self.debugger_args.memory_path)
