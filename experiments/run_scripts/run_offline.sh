#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

task="qa"

modelsize="base"
lr=3e-5
train_bsz=64
pred_bsz=64
num_epochs=30
previous_model="out/mrqa_squad_bart-base_1029_upstream_model/best-model.pt"


warmup=100
max_input_length=888

# 0617v4

ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
logname="retrain_1110_${ns_config}"

upstream_file="data/mrqa_squad/mrqa_squad_train.jsonl"
submission_file="experiments/eval_data/qa/submission_stream.${ns_config}.json"
train_file="experiments/eval_data/qa/offline_retrain.${ns_config}.jsonl"
dev_file="experiments/eval_data/qa/offline_retrain.${ns_config}.dev.jsonl"
output_dir="out/${task}_${ns_config}_offline_retrained_model"

# dev_file="data/mrqa_squad/mrqa_squad_dev.mini.2048.jsonl"
# rm experiments/eval_data/qa/offline_retrain.${ns_config}-preproBartTokenized.json
# TODO: generate the train file 

python semanticdebugger/benchmark_gen/generate_offline_retrainfile.py \
    --upstream_file ${upstream_file} \
    --submission_file ${submission_file} \
    --mixed_offline_file ${train_file} \
    --ratio 0


logfile=logs/offline_retrain.${task}.${logname}.log
python semanticdebugger/cli_bart.py \
        --do_train \
        --output_dir ${output_dir} \
        --model facebook/bart-${modelsize} \
        --checkpoint ${previous_model} \
        --dataset mrqa \
        --train_file ${train_file} \
        --dev_file ${dev_file} \
        --test_file ${dev_file} \
        --learning_rate ${lr} \
        --warmup_steps ${warmup} \
        --train_batch_size ${train_bsz} \
        --predict_batch_size ${pred_bsz} \
        --eval_period 300 \
        --num_train_epochs ${num_epochs} \
        --max_input_length ${max_input_length} \
        --max_output_length 50 \
        --num_beams 3 \
        --append_another_bos  > ${logfile}  2>&1 &

# tail -f logs/${task}.${logname}.log
echo "${logfile}"