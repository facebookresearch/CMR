#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

## Arguments ##
lr=$1
ep=$2
l2w=$3
replay_size=$4
replay_freq=$5
upstream_ratio=$6

## Paths ##
ns_config=$7
task_name=$8
stream_split=$9
stream_id=${10}

seed=${11}
gpu=0
memory_store_rate=1.0


if [ "$task_name" = "qa" ]; then
    upstream_data_path="data/mrqa_squad/mrqa_squad_train.jsonl"
    submission_stream_data="experiments/eval_data/qa/submission_stream.${ns_config}-${stream_split}.json"
    upstream_eval_data="experiments/eval_data/qa/upstream_eval.jsonl"
    heldout_submission_data="experiments/eval_data/qa/heldout_eval.jsonl"
    base_model_path="out/mrqa_squad_bart-base_1029_upstream_model//best-model.pt"
    task_name_arg="mrqa"
elif [ "$task_name" = "nli" ]; then
    upstream_data_path="data/snli/snli_train.jsonl"
    submission_stream_data="experiments/eval_data/nli/submission_stream.${ns_config}-${stream_split}.json"
    upstream_eval_data="experiments/eval_data/nli/upstream_eval.jsonl"
    heldout_submission_data="experiments/eval_data/nli/heldout_eval.jsonl"
    base_model_path="out/snli_bart-base_1109_upstream_model/best-model.pt"
    task_name_arg="nli"
fi

if [ "$stream_split" = "val" ]; then
    use_wandb=False
    max_timecode=100
    save_ckpt_freq=100
    kr_eval_freq=50
    kg_eval_freq=50
elif [ "$stream_split" = "test" ]; then
    use_wandb=True
    max_timecode=100
    save_ckpt_freq=25
    kr_eval_freq=10
    kg_eval_freq=10
fi



prefix="${task_name}_er_lr=${lr}_ep=${ep}_l2w=${l2w}_rs=${replay_size}_rf=${replay_freq}_${ns_config}-${stream_split}[${stream_id}]_seed=${seed}"
log_file="experiments/logs/run_1125_${prefix}.log"
ckpt_dir="experiments/ckpt_dirs/${task_name}/er/${prefix}"
mkdir -p ${ckpt_dir}

echo "Starting ${log_file}."
touch ${log_file}

CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --use_wandb ${use_wandb} \
    --seed $seed --stream_id ${stream_id} \
    --cl_method "er" \
    --task_name ${task_name_arg} \
    --learning_rate ${lr} --num_train_epochs ${ep} \
    --diff_loss_weight ${l2w} \
    --replay_size ${replay_size} --replay_frequency ${replay_freq} \
    --upstream_sample_ratio ${upstream_ratio} \
    --num_adapt_epochs  0 \
    --use_replay_mix \
    --base_model_path ${base_model_path} \
    --num_beams 3 \
    --predict_batch_size 8 \
    --max_timecode ${max_timecode} \
    --kr_eval_freq ${kr_eval_freq} --kr_eval_mode "metric" \
    --kg_eval_freq ${kg_eval_freq} --kg_eval_mode "metric" \
    --prefix ${prefix} \
    --upstream_data_path ${upstream_data_path} \
    --submission_stream_data ${submission_stream_data} \
    --upstream_eval_data ${upstream_eval_data} \
    --heldout_submission_data ${heldout_submission_data} \
    --save_ckpt_freq ${save_ckpt_freq} \
    --ckpt_dir ${ckpt_dir} \
    --init_memory_cache_path "na" \
    --memory_path ${ckpt_dir}/memory_dict.pkl \
    --result_file "experiments/results/${task_name}/${prefix}_result.json" > ${log_file} 
    # 2>&1 
    # &
# tail -f ${log_file}
echo "Finished ${log_file}."
# exit
# exit



