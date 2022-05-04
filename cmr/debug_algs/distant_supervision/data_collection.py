# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
This script is used to get the training data for learning a retriever that can get back the most forgettable examples given a batch of error cases to fix.

Input:
    - The training streams. ---> get the error cases.
    - model.

Output:
    - The pairs between error cases and associated forgettable examples.


Key logic:
    
    - Use the simple_CL method and put it work on the training streams (can be randomly sampled.)
    - For each episode, before and after the error-fixing (continual fine-tuning) step, we record the forgetted the examples.
    

"""

import copy
import pickle
from cmr.debug_algs.cl_utils import _keep_first_answer
from cmr.debug_algs.distant_supervision.ds_utils import create_training_stream, create_training_stream_with_dev
from cmr.debug_algs.index_based.index_manager import RandomMemoryManger
from cmr.debug_algs.index_based.index_utils import get_bart_dual_representation
from cmr.models import run_bart
from cmr.models.utils import set_seeds, trim_batch
import torch

from scipy.stats.stats import describe
from cmr.debug_algs.cl_simple_alg import ContinualFinetuning
import random
from tqdm import tqdm
from cmr.debug_algs import run_lifelong_finetune
from cmr.benchmark_gen import sample_stream_data
import json
from cmr.task_manager.eval_metrics import evaluate_func
from collections import OrderedDict
from operator import getitem

class MiningSupervision(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "simple_ds_mine"
        self.init_model = None
    
    def _check_data_args(self, additional_args):
        pass
    
    def compute_MIR_scores(self, before_model, after_model, examples):
        _examples = _keep_first_answer(examples)
        mlr_data_args = copy.deepcopy(self.data_args)
        mlr_data_args.predict_batch_size = 4    # TODO: set an arg. 
        memory_buffer_loader, _ = self.get_dataloader(
            mlr_data_args, _examples, mode="train", is_training=False) # fix of the order
        
        before_losses = run_bart.inference(
                before_model, memory_buffer_loader, compute_loss=True, loss_only=True, logger=self.logger) 
        after_losses = run_bart.inference(
                after_model, memory_buffer_loader, compute_loss=True, loss_only=True, logger=self.logger)

        MIR_scores = {}
        for example, before_loss, after_loss in zip(examples, before_losses, after_losses): 
            loss_delta = after_loss - before_loss
            MIR_scores[example[2]] = loss_delta  # id, score.
        return MIR_scores

    def get_pos_neg_results(self, examples, scores, positive_size=8, negative_size=8):
        examples_dict = {ex[2]: ex for ex in examples}
        sorted_scores = sorted(scores.items(), key = lambda x:x[1], reverse = True)
        pos_ids = [x[0] for x in sorted_scores[:positive_size]]
        neg_ids = [x[0] for x in sorted_scores[-negative_size:]]
        positive_results = [examples_dict[ex_id] for ex_id in pos_ids]
        negative_results = [examples_dict[ex_id] for ex_id in neg_ids]
        return positive_results, negative_results

    def wrap_supervision(self, before_model, after_model, query_examples, positive_results, negative_results):
        cl_trainer = self
        tokenizer = self.tokenizer
        data_args = copy.deepcopy(self.data_args)
        data_args.predict_batch_size = 4
        # TODO: two options here:



        if self.all_args.long_term_delta:
            # Optional: using the delta versus the init model
            self.logger.info("Using initial model as the before model for computing query vecs.")
            before_model = self.init_model
        
        supervision = {}
        supervision["mode"] = "all_hiddens" if self.all_args.save_all_hiddens else "mean_reps"
        
        supervision["positive"] = {}
        supervision["negative"] = {}
        
        top_examples = []

        if self.all_args.save_all_hiddens:
            supervision["query_before"] = {}
            supervision["query_after"] = {}
            query_hiddens_before = get_bart_dual_representation(cl_trainer, before_model, tokenizer, data_args, query_examples, return_all_hidden=True)    
            query_hiddens_after = get_bart_dual_representation(cl_trainer, after_model, tokenizer, data_args, query_examples, return_all_hidden=True)
            positive_hiddens = get_bart_dual_representation(cl_trainer, self.init_model, tokenizer, data_args, positive_results, return_all_hidden=True)    
            negative_hiddens = get_bart_dual_representation(cl_trainer, self.init_model, tokenizer, data_args, negative_results, return_all_hidden=True)
            for ind, example in enumerate(query_examples):
                supervision["query_before"][example[2]] = {k: v[ind] for k, v in query_hiddens_before.items()}
                supervision["query_after"][example[2]] = {k: v[ind] for k, v in query_hiddens_after.items()}             
            for ind, example in enumerate(positive_results):
                supervision["positive"][example[2]] = {k: v[ind] for k, v in positive_hiddens.items()}
            
            for ind, example in enumerate(negative_hiddens):
                supervision["negative"][example[2]] = {k: v[ind] for k, v in negative_hiddens.items()} 
        else:
            supervision["query"] = {}
            query_vectors_before = get_bart_dual_representation(cl_trainer, before_model, tokenizer, data_args, query_examples)    
            query_vectors_after = get_bart_dual_representation(cl_trainer, after_model, tokenizer, data_args, query_examples)        
            assert len(query_vectors_before) == len(query_vectors_after) == len(query_examples)
            for example, q1, q2 in zip(query_examples, query_vectors_before, query_vectors_after):
                supervision["query"][example[2]] = list(q1) + list(q2) # concat 
            positive_vectors = get_bart_dual_representation(cl_trainer, self.init_model, tokenizer, data_args, positive_results)
            negative_vectors = get_bart_dual_representation(cl_trainer, self.init_model, tokenizer, data_args, negative_results)
            for example, vector in zip(positive_results, positive_vectors):
                supervision["positive"][example[2]] = list(vector)
                top_examples.append(example)
            for example, vector in zip(negative_results, negative_vectors):
                supervision["negative"][example[2]] = list(vector)

        return supervision, top_examples


    def mine_supervision(self, memory_manager=None, all_args=None):
        self.all_args = all_args
        self.logger.info("Start Mining Distant Supervision (as online debugging).")
        
        sub_stream_dataloaders = self.data_eval_loaders

        self.logger.info(f"Number of Batches of Data: {len(sub_stream_dataloaders)}")
        self.logger.info(f"Data Batch Size: {self.data_batch_size};")
        self.timecode = 0

        mined_supervision = []

        for data_eval_loader in tqdm(sub_stream_dataloaders, desc="Mining Supervision from Dynamic Error Stream"):
            episode_data = data_eval_loader.data 
            bug_train_loader, _ = self.get_dataloader(
                self.data_args, episode_data, mode="train")
                # TODO: this is actually not errors for M_t, it is just M_0's errors
            model_copy = copy.deepcopy(self.base_model)
            ############### CORE ###############
            # Fix the bugs by mini-batch based "training"
            self.logger.info(f"Start error-fixing .... Timecode: {self.timecode}")
            self.fix_bugs(bug_train_loader)   # for debugging
            self.logger.info("Start error-fixing .... Done!")
            ############### CORE ###############

            updated_model = self.base_model

            sampled_examples = memory_manager.retrieve_from_memory(sample_size=all_args.mir_buffer_size)
            MIR_scores = self.compute_MIR_scores(model_copy, updated_model, sampled_examples)

            self.timecode += 1
            positive_results, negative_results = self.get_pos_neg_results(sampled_examples, 
                                                                    MIR_scores, positive_size=all_args.positive_size, negative_size=all_args.negative_size)
            supervision, top_examples = self.wrap_supervision(model_copy, updated_model, episode_data, positive_results, negative_results)
            self.logger.info(f"Get an instance for supervision at {self.timecode}")
            mined_supervision.append(supervision)
            memory_manager.store_examples(episode_data)

            # update with the sampled examples 
            self.base_model = model_copy
            self.reset_optimizer()
            mixed_data = episode_data + top_examples
            mixed_bug_train_loader, _ = self.get_dataloader(
                self.data_args, mixed_data, mode="train")
            self.fix_bugs(mixed_bug_train_loader)   # for debugging
            
            # del model_copy

        return mined_supervision

        # if self.debugger_args.save_ckpt_freq:
        #     self._save_base_model()





if __name__ == '__main__':
    parser = run_lifelong_finetune.get_cli_parser()

    
    parser.add_argument("--upstream_data_file", type=str,
                        default="data/mrqa_naturalquestions/mrqa_naturalquestions_train.jsonl",
                        help="the path to upstream data")
    
    parser.add_argument("--upstream_data_prediction_file", type=str,    # by the initial model M_0
                        default="bug_data/mrqa_naturalquestions_train.predictions.jsonl",
                        help="the path to initial model's predictions on the upstream data")

    parser.add_argument("--dev_memory", type=str,    # by the initial model M_0
                        default="exp_results/data_streams/mrqa.nq_train.memory.jsonl",
                        help="the path to initial model's predictions on the upstream data")

    parser.add_argument("--dev_stream", type=str,    # by the initial model M_0
                        default="exp_results/data_streams/mrqa.mixed.data_stream.test.json",
                        help="the path to initial model's predictions on the upstream data")

    parser.add_argument("--output_supervision", type=str,
                        help="the path to save the thread results")

    parser.add_argument('--train_stream_length', type=int, default=100)

    parser.add_argument('--train_stream_episode_size', type=int, default=16)

    parser.add_argument('--init_memory_size', type=int, default=10000)

    parser.add_argument('--num_rounds', type=int, default=1)

    parser.add_argument('--positive_size', type=int, default=8)

    parser.add_argument('--negative_size', type=int, default=8)

    

    parser.add_argument('--mir_buffer_size', type=int, default=256)


    parser.add_argument('--use_dev_stream', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))

    parser.add_argument('--long_term_delta', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))

    parser.add_argument('--save_all_hiddens', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))

    parser.add_argument('--debug_mode', default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    
    args = parser.parse_args()

    # debuggging
    args.cl_method_name = "simple_ds_mine"

    if args.debug_mode:        
        args.use_dev_stream = True
        args.long_term_delta = True

    assert args.cl_method_name == "simple_ds_mine" 


    ## init the useful args ##
    cl_supervision_miner, data_args, base_model_args, debugger_args, logger = run_lifelong_finetune.setup_args(
        args)
    
    setattr(data_args, "replay_stream_json_path", "")


    ## Init the cl_supervision_miner
    cl_supervision_miner.load_base_model(base_model_args)
    cl_supervision_miner.init_model = copy.deepcopy(cl_supervision_miner.base_model)    # maintain M0
    
    
    ## Create Training Stream ##      

    for _rid in range(args.num_rounds):
        logger.info(f"Starting Round {_rid} ....")
        seeds = list(range(100000))
        random.shuffle(seeds)
        selected_seed = seeds[args.seed]    # actually the index
        logger.info(f"Active Seed = {selected_seed}")
        set_seeds(selected_seed)
        if not args.use_dev_stream:
            initial_memory, sampled_train_stream = create_training_stream(args, logger)           
        else:
            initial_memory, sampled_train_stream = create_training_stream_with_dev(args, logger)
        

        ## Init the RandomMemroy module ##    
        memory_manager = RandomMemoryManger(logger) # TODO: try the BART-base one?
        formatted_initial_memory = cl_supervision_miner.data_formatter(initial_memory)
        memory_manager.set_up_initial_memory(formatted_examples=formatted_initial_memory)
        logger.info(f"Initial memory size: {memory_manager.get_memory_size()}")    
        
        cl_supervision_miner.load_data(data_args, given_data_stream=sampled_train_stream)
        cl_supervision_miner.debugger_setup(debugger_args)
        mined_supervision = cl_supervision_miner.mine_supervision(memory_manager, all_args=args)  
        path_to_save = args.output_supervision.replace(".pkl", f"-{_rid}.pkl")
        with open(path_to_save, "wb") as f:
            logger.info(f"Saving {f.name}")
            pickle.dump(mined_supervision, f)
            logger.info(f"Saving {f.name}...Done!")
        logger.info(f"Finished Round {_rid} !")
"""
# debug
index=0
gpu=0
prefix=data_collection_simple_${thread}
log_file=exp_results/supervision_data/logs/run_${prefix}.log
CUDA_VISIBLE_DEVICES=${gpu} python cmr/debug_algs/distant_supervision/data_collection.py \
    --cl_method_name simple_ds_mine \
    --seed ${thread} \
    --output_supervision "exp_results/supervision_data/simple_mir_dm/dm.${thread}.pkl" \
    --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 10 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --replay_stream_json_path "" \
    --upstream_eval_data exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
    --save_ckpt_freq 0     
    > ${log_file} 2>&1 
    
    & 
echo $log_file 

"""
