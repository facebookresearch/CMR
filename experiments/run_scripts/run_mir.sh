#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

lr=$1
ep=$2
replay_size=$3
replay_freq=$4
upstream_ratio=$5
mir_cand_size=$6
seed=42
gpu=0
memory_store_rate=1.0

prefix="QA_mir_lr=${lr}_ep=${ep}_rs=${replay_size}_rf=${replay_freq}_mcs=${mir_cand_size}"
log_file="experiments/logs/run_1031_${prefix}_seed=${seed}.log"
 
log_file=experiments/logs/run_${prefix}.log
echo "Starting ${log_file}."
touch ${log_file}
mkdir experiments/ckpt_dirs/qa/er 
CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --use_wandb True \
    --seed $seed \
    --cl_method "mir" \
    --learning_rate ${lr} --num_train_epochs ${ep} \
    --replay_size ${replay_size} --replay_frequency ${replay_freq} \
    --replay_candidate_size ${mir_cand_size} \
    --upstream_sample_ratio ${upstream_ratio} \
    --num_adapt_epochs  1 \
    --use_replay_mix \
    --base_model_path "out/mrqa_squad_bart-base_1029_upstream_model//best-model.pt" \
    --num_beams 3 \
    --predict_batch_size 32 \
    --max_timecode 100 \
    --kr_eval_freq 5 --kr_eval_mode "metric" \
    --kg_eval_freq 10 --kg_eval_mode "metric" \
    --prefix ${prefix} \
    --upstream_data_path "data/mrqa_squad/mrqa_squad_train.jsonl" \
    --submission_stream_data "experiments/eval_data/qa/dynamic_submission_stream.v1.json" \
    --upstream_eval_data "experiments/eval_data/qa/upstream_eval.v1.jsonl" \
    --heldout_submission_data "experiments/eval_data/qa/heldout_eval.v1.jsonl" \
    --save_ckpt_freq 100 \
    --ckpt_dir "experiments/ckpt_dirs/qa/mir" \
    --init_memory_cache_path "na" \
    --memory_path experiments/ckpt_dirs/qa/mir/${prefix}_memory_dict.pkl \
    --result_file experiments/results/qa/${prefix}_result.json > ${log_file} 
    # 2>&1 
    # &
tail -f ${log_file}
echo "Finished ${log_file}."



