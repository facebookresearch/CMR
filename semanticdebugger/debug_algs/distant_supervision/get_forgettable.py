"""
This script is used to get the training data for learning a retriver that can get back the most forgettable examples given a batch of error cases to fix.

Input:
    - The training streams. ---> get the error cases.
    - model.

Output:
    - The pairs between error cases and associated forgettable examples.


Key logic:
    
    - Use the simple_CL method and put it work on the training streams (can be randomly sampled.)
    - For each episode, before and after the error-fixing (continual fine-tuning) step, we record the forgetted the examples.
    

"""

import pickle
from semanticdebugger.models.utils import trim_batch
import torch

from scipy.stats.stats import describe
from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
import random
from tqdm import tqdm
from semanticdebugger.debug_algs import run_lifelong_finetune
from semanticdebugger.benchmark_gen import sample_stream_data
import json


def sample_sub_stream(data_loaders, target_length):
    assert len(data_loaders) >= target_length
    episode_ids = sorted(random.sample(range(len(data_loaders)), target_length))
    sub_stream = [
        data_loaders[i] for i in episode_ids
    ]
    return sub_stream, episode_ids


class MiningSupervision(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "simple_cl_for_mining_supervision"

    def get_representation(self, example_ids):
        examples = []
        
        for item in self.seen_stream_data:
            if item[2] in example_ids:
                examples.append(item)
        # self.logger.info(f"len(examples) = {len(examples)}")
        if not examples:
            return []
        _, eval_dataloader = self.get_dataloader(
            self.data_args, examples, mode="eval")
        all_tensors = []
        for batch in tqdm(eval_dataloader.dataloader):
            # self.logger.info(f"len(batch)={len(batch)}")
            if self.use_cuda:
                # print(type(batch[0]), batch[0])
                batch = [b.to(torch.device("cuda")) for b in batch]
            pad_token_id = self.tokenizer.pad_token_id
            batch[0], batch[1] = trim_batch(
                batch[0], pad_token_id, batch[1])
            # batch[2], batch[3] = trim_batch(
            #     batch[2], pad_token_id, batch[3])

            input_ids = batch[0]
            attention_mask = batch[1]

            x, encoder_states, all_attentions = self.base_model.model.encoder(
                input_ids, attention_mask)
            # self.logger.info(f"before x.shape={x.shape}")
            x = x[:, 0, :]
            # self.logger.info(f"after x.shape={x.shape}")
            # key_vectors_npy = key_vectors.detach().cpu().numpy()
            vectors = x.detach().cpu().numpy()
            all_tensors += list(vectors)
        # self.logger.info(f"len(all_tensors) = {len(all_tensors)}")
        return all_tensors

    def mine_supervision(self):
        self.logger.info("Start Mining Distant Supervision (as online debugging).")
        # Sample a substream from the training stream.
        # TODO: set up a stream_sample_size = 50
        # sub_stream_dataloaders, episode_ids = sample_sub_stream(self.data_eval_loaders, target_length)
        # self.logger.info(f"episode_ids: {episode_ids}")

        sub_stream_dataloaders = self.data_eval_loaders

        self.logger.info(f"Number of Batches of Data: {len(sub_stream_dataloaders)}")
        self.logger.info(f"Data Batch Size: {self.data_batch_size};")
        self.timecode = 0

        mined_supervision = []
        self.overall_errors = []
        self.seen_stream_data = []

        for data_eval_loader in tqdm(sub_stream_dataloaders, desc="Mining Supervision from Dynamic Error Stream"):

            result_dict = {"timecode": self.timecode}   # start with 0

            # self._replay_based_eval(result_dict)
            bug_train_loader = self._get_dynamic_errors(data_eval_loader, result_dict)

            # TODO: save the model-status related information about the exampels.
            # e.g., the adapter weights?
            # - to build the query vector and the memory.

            ############### CORE ###############
            # Fix the bugs by mini-batch based "training"
            self.logger.info(f"Start bug-fixing .... Timecode: {self.timecode}")
            self.fix_bugs(bug_train_loader)   # for debugging
            self.logger.info("Start bug-fixing .... Done!")
            ############### CORE ###############

            self._log_episode_result(result_dict, data_eval_loader)
            self.timecode += 1

            supervision = {}
            supervision["error_ids"] = result_dict["fixed_ids"] + result_dict["unfixed_ids"]
            supervision["forgotten_examples"] = result_dict["forgotten_examples"]
            supervision["unforgettable_ids"] = result_dict["retained_ids"]
            supervision["fixed_ids"] = result_dict["fixed_ids"]

            supervision["fixed_error_vectors"] = self.get_representation(supervision["fixed_ids"])
            supervision["forgotten_examples_vectors"] = self.get_representation(
                [item["id"] for item in supervision["forgotten_examples"]])
            supervision["unforgettable_examples_vectors"] = self.get_representation(
                supervision["unforgettable_ids"])

            # supervision["model_weights"] = {}

            mined_supervision.append(supervision)

            self.seen_stream_data += data_eval_loader.data

        return mined_supervision

        # if self.debugger_args.save_all_ckpts:
        #     self._save_base_model()


if __name__ == '__main__':
    parser = run_lifelong_finetune.get_cli_parser()

    parser.add_argument("--output_supervision", type=str,
                        help="the path to save the thread results")
    parser.add_argument('--num_rounds', type=int, default=6)
    parser.add_argument('--stream_len', type=int, default=100)

    args = parser.parse_args()

    args.seed += 42  # 42 + 0...n_threads

    assert args.cl_method_name == "simple_cl_for_mining_supervision"

    debugging_alg, data_args, base_model_args, debugger_args, logger = run_lifelong_finetune.setup_args(
        args)

    setattr(data_args, "data_stream_json_path", args.data_stream_json_path)
    setattr(data_args, "replay_stream_json_path", args.replay_stream_json_path)

    with open(data_args.data_stream_json_path) as f:
        data_stream = json.load(f)

    data_pool = []
    for episode in data_stream:
        data_pool += episode

    logger.info(f"len(data_pool): {len(data_pool)}")
    rounds = args.num_rounds
    batch_size = 32
    target_substream_length = args.stream_len
    all_mined_supervision = []
    for _round_id in tqdm(range(rounds), desc="Sample and get results."):
        sampled_stream = sample_stream_data.get_data_stream_with_replacement(
            data_pool, batch_size=batch_size, num_batches=target_substream_length)
        debugging_alg.load_data(data_args, given_data_stream=sampled_stream)
        debugging_alg.load_base_model(base_model_args)
        debugging_alg.debugger_setup(debugger_args)
        mined_supervision = debugging_alg.mine_supervision()
        all_mined_supervision += mined_supervision
        del debugging_alg.base_model
        
    # with open(args.output_supervision, "w") as f:
    #     json.dump(all_mined_supervision, f)
    
    with open(args.output_supervision, "wb") as f:
        pickle.dump(all_mined_supervision, f)

"""
n_threads=8
n_gpus=8
start_gpuid=0
for (( thread=0; thread<${n_threads}; thread++ ))
do 

    prefix=nq_dev_0812_wr_mined_supervision_from_train_${thread}
    log_file=exp_results/supervision_data/run_${prefix}.log
    echo ${log_file}
    touch ${log_file}
    gpu=$(($start_gpuid + $thread % $n_gpus ))
    echo $thread, $gpu
    CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/distant_supervision/get_forgettable.py \
        --cl_method_name simple_cl_for_mining_supervision \
        --num_rounds 10 --stream_len 100 \
        --seed ${thread} \
        --output_supervision "exp_results/supervision_data/error_forget_pairs.${thread}.pkl" \
        --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 10 \
        --prefix ${prefix} \
        --stream_mode dynamic \
        --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.train.wr.json \
        --replay_stream_json_path "" \
        --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
        --save_all_ckpts 0 \
        --result_file exp_results/supervision_data/${prefix}_result.json > ${log_file} 2>&1 & 
    echo $log_file
done



python semanticdebugger/benchmark_gen/merge_json_file.py \
    --input_file_pattern exp_results/supervision_data/error_forget_pairs.#.json \
    --range "range(8)" \
    --output_file exp_results/supervision_data/error_forget_pairs.json
"""


"""
# debug
prefix=nq_dev_0816_wr_mined_supervision_from_train_${thread}
log_file=exp_results/supervision_data/run_${prefix}.log
thread=0
gpu=0
CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/distant_supervision/get_forgettable.py \
    --num_rounds 1 --stream_len 10 \
    --cl_method_name simple_cl_for_mining_supervision \
    --seed ${thread} \
    --output_supervision "exp_results/supervision_data/error_forget_pairs.${thread}.npy" \
    --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 10 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.train.wr.json \
    --replay_stream_json_path "" \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
    --save_all_ckpts 0 \
    --result_file exp_results/supervision_data/${prefix}_result.json.tmp 
    
    > ${log_file} 2>&1 & 
echo $log_file 


"""
