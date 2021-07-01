prefix=nq_dev_0625_v1
log_file=logs/tmp/online_debug_${prefix}.log
echo ${log_file}
python semanticdebugger/debug_algs/run_continual_finetune.py \
    --learning_rate 1e-5 --num_train_epochs 3 \
    --prefix ${prefix} \
    --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
tail -f ${log_file}