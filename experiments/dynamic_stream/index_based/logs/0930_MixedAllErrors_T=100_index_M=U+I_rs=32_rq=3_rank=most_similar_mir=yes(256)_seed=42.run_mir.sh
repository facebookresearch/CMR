#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

# Index-based Replay

max_timecode=$1
replay_size=$2
replay_freq=$3
rank_method=$4
use_mir=$5
mir_cand_size=$6
seed=$7
gpu=0 
# declare -a seeds=("42" "0212" "1213")
# for seed in "${seeds[@]}"
# do
memory_store_rate=1.0
prefix="0930_MixedAllErrors_T=${max_timecode}_index_M=U+I_rs=${replay_size}_rq=${replay_freq}_rank=${rank_method}_mir=${use_mir}(${mir_cand_size})_seed=${seed}"
log_file=exp_results/dynamic_stream/index_based/logs/run_${prefix}.log
mkdir exp_results/dynamic_stream/index_based/ckpt_dir/${prefix}_ckpts/
tmp_script_copy=exp_results/dynamic_stream/index_based/logs/${prefix}.run_mir.sh
tmp_code_copy=exp_results/dynamic_stream/index_based/logs/${prefix}.cl_indexed_alg.pybackup
cp semanticdebugger/debug_algs/index_based/cl_indexed_alg.py $tmp_code_copy
cp exp_results/dynamic_stream/index_based/run_index.sh $tmp_script_copy

echo ${log_file}


# M=U+I --use_sampled_upstream \

CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --seed $seed \
    --max_timecode ${max_timecode} \
    --cl_method_name "index_cl" \
    --use_mir ${use_mir} \
    --replay_candidate_size ${mir_cand_size} \
    --mir_abalation_args "none" \
    --index_rank_method ${rank_method} \
    --memory_key_encoder "facebook/bart-base" \
    --memory_store_rate ${memory_store_rate} \
    --num_adapt_epochs 0 \
    --use_replay_mix \
    --replay_size ${replay_size} --replay_frequency ${replay_freq} \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --use_sampled_upstream \
    --sampled_upstream_json_path exp_results/data_streams/mrqa.nq_train.memory.jsonl \
    --data_stream_json_path exp_results/data_streams/mrqa.mixed.data_stream.test.json \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa.mixed.upstream_eval.jsonl \
    --replay_stream_json_path "" \
    --save_all_ckpts 0 \
    --init_memory_cache_path "exp_results/data_streams/init_memory.pkl" \
    --memory_path "exp_results/dynamic_stream/index_based/ckpt_dir/${prefix}_ckpts/memory_dict.pkl" \
    --overtime_ckpt_dir exp_results/dynamic_stream/index_based/ckpt_dir/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/index_based/results/${prefix}_result.json > ${log_file} 2>&1
    #  2>&1 &
# gpu=$((gpu+1))
# done


# --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.wr.wpara.json \
# --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \