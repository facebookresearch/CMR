#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/


### Simple CL###

seed=$1
gpu=0 
# declare -a seeds=("42" "0212" "1213")
# for seed in "${seeds[@]}"
# do

prefix="0926_MixedAllError_T=100_simple_ep=10_seed=${seed}"
log_file=exp_results/dynamic_stream/memory_based/logs/run_${prefix}.log
mkdir exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/
tmp_script_copy=exp_results/dynamic_stream/memory_based/logs/${prefix}.run_simple.sh
cp exp_results/dynamic_stream/cl_simple/run_simple.sh $tmp_script_copy

echo ${log_file}
 
CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --seed $seed \
    --max_timecode 100 \
    --cl_method_name "simple_cl" \
    --learning_rate 3e-5 --num_train_epochs 10 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --data_stream_json_path exp_results/data_streams/mrqa.mixed.data_stream.test.json \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa.mixed.upstream_eval.jsonl \
    --replay_stream_json_path "" \
    --save_all_ckpts 0 \
    --overtime_ckpt_dir exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/memory_based/results/${prefix}_result.json > ${log_file}
    #  2>&1 &
# gpu=$((gpu+1))
# done


# --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.wr.wpara.json \
# --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \