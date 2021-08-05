lambda=500
prefix=nq_dev_0804v2_dynamic_ewc
log_file=exp_results/dynamic_stream/online_ewc/run_${prefix}.log
touch ${log_file}
echo ${log_file}
CUDA_VISIBLE_DEVICES=6,7 python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --cl_method_name "online_ewc" \
    --ewc_lambda ${lambda} --ewc_gamma 1 \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --max_timecode 30 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.json \
    --replay_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.replay_stream.test.json \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
    --save_all_ckpts 0 \
    --overtime_ckpt_dir exp_results/dynamic_stream/online_ewc/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/online_ewc/${prefix}_result.json > ${log_file} 2>&1 & 
# tail -f ${log_file}

