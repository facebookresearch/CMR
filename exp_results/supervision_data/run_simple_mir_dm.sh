#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

index=$1
num_rounds=$2
gpu=0
pprefix=1014v2_hiddens_dm_simple
prefix=${pprefix}_${index}
log_file=exp_results/supervision_data/logs/run_${prefix}.log
mkdir exp_results/supervision_data/${pprefix}/
CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/distant_supervision/data_collection.py \
    --save_all_hiddens True \
    --base_model_path "out/mrqa_naturalquestions_bart-base_1011/best-model.pt" \
    --upstream_data_prediction_file "bug_data/1011_mrqa_naturalquestions_train.meta.predictions.json" \
    --cl_method_name "simple_ds_mine" \
    --seed ${index} \
    --max_input_length 512 --max_output_length 64 \
    --num_rounds ${num_rounds} \
    --output_supervision "exp_results/supervision_data/${pprefix}/dm.${index}.pkl" \
    --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 4 --predict_batch_size 4 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --replay_stream_json_path "" \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
    --save_all_ckpts 0  > ${log_file} 2>&1
# echo $log_file 


# --base_model_path "out/mrqa_naturalquestions_bart-base_1011/best-model.pt" \
# --upstream_data_prediction_file "bug_data/1011_mrqa_naturalquestions_train.meta.predictions.json" \

