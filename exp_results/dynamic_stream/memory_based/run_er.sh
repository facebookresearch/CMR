#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

# Experience Replay

max_timecode=$1
replay_size=$2
replay_freq=$3
seed=$4
gpu=0 
# declare -a seeds=("42" "0212" "1213")
# for seed in "${seeds[@]}"
# do
memory_store_rate=1.0
prefix="0927_MixedAllErrors_T=100_er_M=U+I_rs=32_rq=3_seed=${seed}"
log_file=exp_results/dynamic_stream/memory_based/logs/run_${prefix}.log
mkdir exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/
tmp_script_copy=exp_results/dynamic_stream/memory_based/logs/${prefix}.run_mir.sh
tmp_code_copy=exp_results/dynamic_stream/memory_based/logs/${prefix}.cl_mbcl_alg.py
cp semanticdebugger/debug_algs/cl_mbcl_alg.py $tmp_code_copy
cp exp_results/dynamic_stream/memory_based/run_er.sh $tmp_script_copy

echo ${log_file}


# M=U+I --use_sampled_upstream \

CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --seed $seed \
    --max_timecode ${max_timecode} \
    --cl_method_name "er" \
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
    --memory_path exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/memory_dict.pkl \
    --init_memory_cache_path "na" \
    --overtime_ckpt_dir exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/memory_based/results/${prefix}_result.json > ${log_file}
    #  2>&1 &
# gpu=$((gpu+1))
# done


# --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.wr.wpara.json \
# --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \