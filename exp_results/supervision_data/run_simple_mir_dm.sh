#!/bin/bash
source ~/.bashrc
conda activate bartqa 
cd ~/SemanticDebugger/

index=0
gpu=0
prefix=data_collection_simple_${index}
log_file=exp_results/supervision_data/logs/run_${prefix}.log
CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/distant_supervision/data_collection.py \
    --cl_method_name "simple_ds_mine" \
    --seed ${index} \
    --output_supervision "exp_results/supervision_data/simple_mir_dm/dm.${index}.pkl" \
    --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 10 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --replay_stream_json_path "" \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
    --save_all_ckpts 0  > ${log_file} 
    # 2 >&1
# echo $log_file 
