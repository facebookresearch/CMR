#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

## Paths ##
ns_config=$4
upstream_data_path="data/mrqa_squad/mrqa_squad_train.jsonl"
submission_stream_data="experiments/eval_data/qa/submission_stream.${ns_config}.json"
upstream_eval_data="experiments/eval_data/qa/upstream_eval.jsonl"
heldout_submission_data="experiments/eval_data/qa/heldout_eval.jsonl"

## Args ##
seed=$1
lr=$2
ep=$3
gpu=0

prefix="QA_simplecl_lr=${lr}_ep=${ep}_${ns_config}"
ckpt_dir="experiments/ckpt_dirs/qa/er/${prefix}"
mkdir -p ${ckpt_dir}


log_file="experiments/logs/run_1107_${prefix}_seed=${seed}.log"
echo "Starting ${log_file}."
touch ${log_file}
mkdir experiments/ckpt_dirs/qa/simplecl

CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --use_wandb True \
    --seed $seed \
    --cl_method "simple_cl" \
    --learning_rate ${lr} --num_train_epochs ${ep} \
    --base_model_path "out/mrqa_squad_bart-base_1029_upstream_model//best-model.pt" \
    --num_beams 3 \
    --predict_batch_size 48 \
    --max_timecode 100 \
    --kr_eval_freq 10 --kr_eval_mode "metric" \
    --kg_eval_freq 10 --kg_eval_mode "metric" \
    --prefix ${prefix} \
    --submission_stream_data ${submission_stream_data} \
    --upstream_eval_data ${upstream_eval_data} \
    --heldout_submission_data ${heldout_submission_data} \
    --save_ckpt_freq 10 \
    --ckpt_dir ${ckpt_dir} \
    --result_file experiments/results/qa/${prefix}_result.json > ${log_file} 
    # 2>&1 
    # &
# tail -f ${log_file}
echo "Finished ${log_file}."
# exit
# exit

