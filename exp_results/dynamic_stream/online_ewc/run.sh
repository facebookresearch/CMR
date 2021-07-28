lambda=500
prefix=nq_dev_0723_dynamic_ewc_l${lambda}_50
log_file=exp_results/dynamic_stream/online_ewc/run_${prefix}.log
echo ${log_file}
CUDA_VISIBLE_DEVICES=2,3 python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --cl_method_name "online_ewc" \
    --ewc_lambda ${lambda} --ewc_gamma 1 \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --max_timecode 50 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --data_stream_json_path bug_data/mrqa_naturalquestions_dev.data_stream.test.json \
    --save_all_ckpts 1 \
    --overtime_ckpt_dir exp_results/dynamic_stream/online_ewc/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/online_ewc/${prefix}_result.json > ${log_file} 2>&1 & 
tail -f ${log_file}

