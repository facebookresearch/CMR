#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

## Args ##
seed=42

## Paths ##
ns_config=$1
task_name=$2

if [ "$task_name" = "qa" ]; then
    upstream_data_path="data/mrqa_squad/mrqa_squad_train.jsonl"
    submission_stream_data="experiments/eval_data/qa/submission_stream.${ns_config}.json"
    upstream_eval_data="experiments/eval_data/qa/upstream_eval.jsonl"
    heldout_submission_data="experiments/eval_data/qa/heldout_eval.jsonl"
    base_model_path="out/mrqa_squad_bart-base_1029_upstream_model//best-model.pt"
    task_name_arg="mrqa"
elif [ "$task_name" = "nli" ]; then
    upstream_data_path="data/snli/snli_train.jsonl"
    submission_stream_data="experiments/eval_data/nli/submission_stream.${ns_config}.json"
    upstream_eval_data="experiments/eval_data/nli/upstream_eval.jsonl"
    heldout_submission_data="experiments/eval_data/nli/heldout_eval.jsonl"
    base_model_path="out/snli_bart-base_1109_upstream_model/best-model.pt"
    task_name_arg="nli"
fi

gpu=0
prefix="${task_name}_nonecl_${ns_config}"

ckpt_dir="experiments/ckpt_dirs/${task_name}/er/${prefix}"
mkdir -p ${ckpt_dir}

log_file="experiments/logs/run_1107_${prefix}_seed=${seed}.log"
echo "Starting ${log_file}."
touch ${log_file} 

CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --use_wandb True \
    --seed $seed \
    --task_name ${task_name_arg} \
    --cl_method "none_cl" \
    --base_model_path ${base_model_path} \
    --num_beams 3 \
    --learning_rate 0 --num_train_epochs 0 \
    --predict_batch_size 64 \
    --max_timecode 100 \
    --kr_eval_freq 10 --kr_eval_mode "metric" \
    --kg_eval_freq 10 --kg_eval_mode "metric" \
    --prefix ${prefix} \
    --submission_stream_data ${submission_stream_data} \
    --upstream_eval_data ${upstream_eval_data} \
    --heldout_submission_data ${heldout_submission_data} \
    --save_ckpt_freq 10 \
    --ckpt_dir ${ckpt_dir} \
    --result_file "experiments/results/${task_name}/${prefix}_result.json" > ${log_file} 
    # 2>&1 
    # &
# tail -f ${log_file}
echo "Finished ${log_file}."
# exit
# exit