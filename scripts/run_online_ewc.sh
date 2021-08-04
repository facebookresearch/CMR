# prefix=nq_dev_0625_v1
# log_file=logs/tmp/online_debug_${prefix}.log
# echo ${log_file}
# python semanticdebugger/debug_algs/run_lifelong_finetune.py \
#     --learning_rate 1e-5 --num_train_epochs 3 \
#     --prefix ${prefix} \
#     --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
# tail -f ${log_file}

prefix=nq_dev_0708_ewc
log_file=logs/tmp/online_debug_${prefix}.log
echo ${log_file}
python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --cl_method_name "online_ewc" \
    --ewc_lambda 0.5 --ewc_gamma 1 \
    --learning_rate 3e-5 --num_train_epochs 3 \
    --prefix ${prefix} \
    --save_all_ckpts 1 \
    --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
    --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
tail -f ${log_file}