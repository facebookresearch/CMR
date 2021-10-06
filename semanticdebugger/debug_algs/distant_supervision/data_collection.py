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
from semanticdebugger.debug_algs.cl_utils import _keep_first_answer
from semanticdebugger.debug_algs.distant_supervision.ds_utils import create_training_stream
from semanticdebugger.debug_algs.index_based.index_manager import RandomMemoryManger
from semanticdebugger.debug_algs.index_based.index_utils import get_bart_dual_representation
from semanticdebugger.models import run_bart
from semanticdebugger.models.utils import set_seeds, trim_batch
import torch

from scipy.stats.stats import describe
from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
import random
from tqdm import tqdm
from semanticdebugger.debug_algs import run_lifelong_finetune
from semanticdebugger.benchmark_gen import sample_stream_data
import json
from semanticdebugger.task_manager.eval_metrics import evaluate_func
from collections import OrderedDict
from operator import getitem

class MiningSupervision(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "simple_data_collection"
        self.init_model = None
    
    def _check_data_args(self, additional_args):
        pass
    
    def compute_MIR_scores(self, before_model, after_model, examples):
        _examples = _keep_first_answer(examples)
        mlr_data_args = copy.deepcopy(self.data_args)
        mlr_data_args.predict_batch_size = 8    # TODO: set an arg. 
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
        data_args.predict_batch_size = 8
        query_vectors_before = get_bart_dual_representation(cl_trainer, before_model, tokenizer, data_args, query_examples)
        query_vectors_after = get_bart_dual_representation(cl_trainer, after_model, tokenizer, data_args, query_examples)
        assert len(query_vectors_before) == len(query_vectors_after) == len(query_examples)
        supervision = {}
        supervision["query"] = {}
        supervision["positive"] = {}
        supervision["negative"] = {}

        for example, q1, q2 in zip(query_examples, query_vectors_before, query_vectors_after):
            supervision["query"][example[2]] = list(q1) + list(q2) # concat 

        positive_vectors = get_bart_dual_representation(cl_trainer, self.init_model, tokenizer, data_args, positive_results)
        negative_vectors = get_bart_dual_representation(cl_trainer, self.init_model, tokenizer, data_args, negative_results)
        for example, vector in zip(positive_results, positive_vectors):
            supervision["positive"][example[2]] = list(vector)

        for example, vector in zip(negative_results, negative_vectors):
            supervision["negative"][example[2]] = list(vector)
        return supervision


    def mine_supervision(self, memory_manager=None):
        self.logger.info("Start Mining Distant Supervision (as online debugging).")
        
        sub_stream_dataloaders = self.data_eval_loaders

        self.logger.info(f"Number of Batches of Data: {len(sub_stream_dataloaders)}")
        self.logger.info(f"Data Batch Size: {self.data_batch_size};")
        self.timecode = 0

        mined_supervision = []

        for data_eval_loader in tqdm(sub_stream_dataloaders, desc="Mining Supervision from Dynamic Error Stream"):

            result_dict = {"timecode": self.timecode}   # start with 0

            # self._replay_based_eval(result_dict)
            # bug_train_loader = self._get_dynamic_errors(data_eval_loader, result_dict)

            episode_data = data_eval_loader.data 
            bug_train_loader, _ = self.get_dataloader(
                self.data_args, episode_data, mode="train")
                # TODO: this is actually not errors for M_t, it is just M_0's errors
            model_copy = copy.deepcopy(self.base_model)

            ############### CORE ###############
            # Fix the bugs by mini-batch based "training"
            self.logger.info(f"Start bug-fixing .... Timecode: {self.timecode}")
            self.fix_bugs(bug_train_loader)   # for debugging
            self.logger.info("Start bug-fixing .... Done!")
            ############### CORE ###############

            updated_model = self.base_model

            sampled_examples = memory_manager.retrieve_from_memory(sample_size=256) # TODO: set an arg.
            MIR_scores = self.compute_MIR_scores(model_copy, updated_model, sampled_examples)

            self.timecode += 1
            positive_results, negative_results = self.get_pos_neg_results(sampled_examples, 
                                                                    MIR_scores, positive_size=8, negative_size=8)
            

            supervision = self.wrap_supervision(model_copy, updated_model, episode_data, positive_results, negative_results)
            # supervision["error_ids"] = result_dict["fixed_ids"] + result_dict["unfixed_ids"]
            # supervision["forgotten_examples"] = result_dict["forgotten_examples"]
            # supervision["unforgettable_ids"] = result_dict["retained_ids"]
            # supervision["fixed_ids"] = result_dict["fixed_ids"]
            # supervision["model_weights"] = {}

            mined_supervision.append(supervision)
            memory_manager.store_examples(episode_data)

        return mined_supervision

        # if self.debugger_args.save_all_ckpts:
        #     self._save_base_model()





if __name__ == '__main__':
    parser = run_lifelong_finetune.get_cli_parser()

    parser.add_argument("--upstream_data_file", type=str,
                        default="data/mrqa_naturalquestions/mrqa_naturalquestions_train.jsonl",
                        help="the path to upstream data")
    
    parser.add_argument("--upstream_data_prediction_file", type=str,    # by the initial model M_0
                        default="bug_data/mrqa_naturalquestions_train.predictions.jsonl",
                        help="the path to initial model's predictions on the upstream data")

    parser.add_argument("--output_supervision", type=str,
                        help="the path to save the thread results")

    parser.add_argument('--train_stream_length', type=int, default=100)

    parser.add_argument('--train_stream_episode_size', type=int, default=16)

    parser.add_argument('--init_memory_size', type=int, default=10000)

    args = parser.parse_args()


    assert args.cl_method_name == "simple_data_collection" 

    ## Create Training Stream ##
    set_seeds(args.seed)
    initial_memory, sampled_train_stream = create_training_stream(args)
        
    
    ## init the useful args ##
    cl_supervision_miner, data_args, base_model_args, debugger_args, logger = run_lifelong_finetune.setup_args(
        args)
    
    setattr(data_args, "replay_stream_json_path", "")

    ## Init the RandomMemroy module ##    
    memory_manager = RandomMemoryManger(logger) # TODO: try the BART-base one?
    formatted_initial_memory = cl_supervision_miner.data_formatter(initial_memory)
    memory_manager.set_up_initial_memory(formatted_examples=formatted_initial_memory)
    logger.info(f"Initial memory size: {memory_manager.get_memory_size()}")    
    


    ## Init the cl_supervision_miner
    cl_supervision_miner.load_data(data_args, given_data_stream=sampled_train_stream)
    cl_supervision_miner.load_base_model(base_model_args)

    cl_supervision_miner.init_model = copy.deepcopy(cl_supervision_miner.base_model)    # maintain M0
    cl_supervision_miner.debugger_setup(debugger_args)
    mined_supervision = cl_supervision_miner.mine_supervision(memory_manager)  
    with open(args.output_supervision, "wb") as f:
        pickle.dump(mined_supervision, f)

"""
n_threads=8
n_gpus=8
start_gpuid=0
for (( thread=0; thread<${n_threads}; thread++ ))
do 
    prefix=nq_dev_0812_wr_mined_supervision_from_train_${thread}
    log_file=exp_results/supervision_data/logs/run_${prefix}.log
    echo ${log_file}
    touch ${log_file}
    gpu=$(($start_gpuid + $thread % $n_gpus ))
    echo $thread, $gpu
    CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/distant_supervision/data_collection.py \
        --cl_method_name simple_data_collection \
        --num_rounds 10 --stream_len 100 \
        --seed ${thread} \
        --output_supervision "exp_results/supervision_data/error_forget_pairs.${thread}.pkl" \
        --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 10 \
        --prefix ${prefix} \
        --stream_mode dynamic \
        --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.train.wr.json \
        --replay_stream_json_path "" \
        --pass_pool_jsonl_path exp_results/data_streams/mrqa.mixed.upstream_eval.jsonl \
        --save_all_ckpts 0 \
        --result_file exp_results/supervision_data/results/${prefix}_result.json > ${log_file} 2>&1 & 
    echo $log_file
done



python semanticdebugger/benchmark_gen/merge_json_file.py \
    --input_file_pattern exp_results/supervision_data/error_forget_pairs.#.json \
    --range "range(8)" \
    --output_file exp_results/supervision_data/error_forget_pairs.json
"""


"""
# debug
thread=0
gpu=1
prefix=data_collection_simple_${thread}
log_file=exp_results/supervision_data/logs/run_${prefix}.log
CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/distant_supervision/data_collection.py \
    --cl_method_name simple_data_collection \
    --seed ${thread} \
    --output_supervision "exp_results/supervision_data/error_forget_pairs.${thread}.npy" \
    --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 10 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --replay_stream_json_path "" \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
    --save_all_ckpts 0     
    > ${log_file} 2>&1 & 
echo $log_file 


"""
