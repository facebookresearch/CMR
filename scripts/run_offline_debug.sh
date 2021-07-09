# prefix=nq_dev_0709_offlinedebug_3e-5_e5
# log_file=logs/tmp/online_debug_${prefix}.log
# echo ${log_file}
# CUDA_VISIBLE_DEVICES=0,1 python semanticdebugger/debug_algs/run_lifelong_finetune.py \
#     --cl_method_name "offline_debug" \
#     --learning_rate 3e-5 --num_train_epochs 5 \
#     --prefix ${prefix} \
#     --save_all_ckpts 1 \
#     --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
#     --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 


prefix=nq_dev_0709_offlinedebug_withup_3e-5_e5
log_file=logs/tmp/online_debug_${prefix}.log
echo ${log_file}
CUDA_VISIBLE_DEVICES=0,1 python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --cl_method_name "offline_debug" \
    --use_sampled_upstream \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --prefix ${prefix} \
    --save_all_ckpts 1 \
    --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
    --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 