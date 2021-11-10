#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

## Args ##
lr=$1
ep=$2
replay_size=$3
replay_freq=$4
upstream_ratio=$5
mir_cand_size=$6
mir_abalation_args=$7   # none, largest_beforeloss, largest_afterloss

memory_store_rate=1.0
seed=42

## Paths ##
ns_config=$8
task_name=$9


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


prefix="${task_name}_mir_lr=${lr}_ep=${ep}_rs=${replay_size}_rf=${replay_freq}_mcs=${mir_cand_size}_${mir_abalation_args}_${ns_config}"
ckpt_dir="experiments/ckpt_dirs/${task_name}/mir/${prefix}"
mkdir -p ${ckpt_dir}

log_file="experiments/logs/run_1109_${prefix}_seed=${seed}.log"
echo "Starting ${log_file}."
touch ${log_file}


CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --use_wandb True \
    --seed $seed \
    --cl_method "mir" \
    --mir_abalation_args ${mir_abalation_args} \
    --learning_rate ${lr} --num_train_epochs ${ep} \
    --replay_size ${replay_size} --replay_frequency ${replay_freq} \
    --replay_candidate_size ${mir_cand_size} \
    --upstream_sample_ratio ${upstream_ratio} \
    --num_adapt_epochs  1 \
    --use_replay_mix \
    --base_model_path ${base_model_path} \
    --num_beams 3 \
    --predict_batch_size 32 \
    --max_timecode 100 \
    --kr_eval_freq 10 --kr_eval_mode "metric" \
    --kg_eval_freq 10 --kg_eval_mode "metric" \
    --prefix ${prefix} \
    --upstream_data_path ${upstream_data_path} \
    --submission_stream_data ${submission_stream_data} \
    --upstream_eval_data ${upstream_eval_data} \
    --heldout_submission_data ${heldout_submission_data} \
    --save_ckpt_freq 10 \
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


