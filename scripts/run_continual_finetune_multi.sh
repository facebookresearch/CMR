prefix=nq_dev_0706_1e-5_e3
log_file=logs/tmp/online_debug_${prefix}.log
echo ${log_file}
CUDA_VISIBLE_DEVICES=0,1 python semanticdebugger/debug_algs/run_continual_finetune.py \
    --learning_rate 1e-5 --num_train_epochs 3 \
    --prefix ${prefix} \
    --save_all_ckpts 1 \
    --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
    --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 

prefix=nq_dev_0706_3e-5_e3
log_file=logs/tmp/online_debug_${prefix}.log
echo ${log_file}
CUDA_VISIBLE_DEVICES=2,3 python semanticdebugger/debug_algs/run_continual_finetune.py \
    --learning_rate 3e-5 --num_train_epochs 3 \
    --prefix ${prefix} \
    --save_all_ckpts 1 \
    --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
    --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 

prefix=nq_dev_0706_1e-5_e5
log_file=logs/tmp/online_debug_${prefix}.log
echo ${log_file}
CUDA_VISIBLE_DEVICES=4,5 python semanticdebugger/debug_algs/run_continual_finetune.py \
    --learning_rate 1e-5 --num_train_epochs 5 \
    --prefix ${prefix} \
    --save_all_ckpts 1 \
    --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
    --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 

prefix=nq_dev_0706_3e-5_e5
log_file=logs/tmp/online_debug_${prefix}.log
echo ${log_file}
CUDA_VISIBLE_DEVICES=6,7 python semanticdebugger/debug_algs/run_continual_finetune.py \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --prefix ${prefix} \
    --save_all_ckpts 1 \
    --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
    --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
