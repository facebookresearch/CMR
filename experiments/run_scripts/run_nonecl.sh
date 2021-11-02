#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

seed=42
gpu=0
prefix="QA_nonecl"
log_file="experiments/logs/run_1030_${prefix}_seed=${seed}.log"
echo "Starting ${log_file}."
touch ${log_file}
mkdir experiments/ckpt_dirs/qa/nonecl

CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --use_wandb True \
    --seed $seed \
    --cl_method "none_cl" \
    --base_model_path "out/mrqa_squad_bart-base_1029_upstream_model//best-model.pt" \
    --num_beams 3 \
    --predict_batch_size 64 \
    --max_timecode 100 \
    --kr_eval_freq 5 \
    --kr_eval_mode "metric" \
    --kg_eval_freq 10 \
    --kg_eval_mode "metric" \
    --prefix ${prefix} \
    --submission_stream_data "experiments/eval_data/qa/dynamic_submission_stream.v1.json" \
    --upstream_eval_data "experiments/eval_data/qa/upstream_eval.v1.jsonl" \
    --heldout_submission_data "experiments/eval_data/qa/heldout_eval.v1.jsonl" \
    --save_ckpt_freq 30 \
    --ckpt_dir "experiments/ckpt_dirs/qa/nonecl" \
    --result_file experiments/results/qa/${prefix}_result.json > ${log_file} 
    # 2>&1 
    # &
tail -f ${log_file}
echo "Finished ${log_file}."
