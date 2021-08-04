lambda=500
prefix=nq_dev_0729_dynamic_ewc_l${lambda}
log_file=exp_results/dynamic_stream/online_ewc/run_${prefix}.log
echo ${log_file}
CUDA_VISIBLE_DEVICES=3,4 python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --cl_method_name "online_ewc" \
    --ewc_lambda ${lambda} --ewc_gamma 1 \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --max_timecode 30 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.json \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
    --save_all_ckpts 0 \
    --overtime_ckpt_dir exp_results/dynamic_stream/online_ewc/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/online_ewc/${prefix}_result.json > ${log_file} 2>&1 & 
tail -f ${log_file}
