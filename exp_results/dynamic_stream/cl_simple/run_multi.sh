prefix=nq_dev_0723_dynamic_simplecl_50
log_file=exp_results/dynamic_stream/cl_simple/run_${prefix}.log
echo ${log_file}
python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --max_timecode 50 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --data_stream_json_path bug_data/mrqa_naturalquestions_dev.data_stream.test.json \
    --save_all_ckpts 1 \
    --overtime_ckpt_dir exp_results/dynamic_stream/cl_simple/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/cl_simple/${prefix}_result.json > ${log_file} 2>&1 & 
tail -f ${log_file}