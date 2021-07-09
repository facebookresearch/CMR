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


prefix=nq_dev_0709_simplereplay_rsz10_3e-5_e5
log_file=logs/tmp/online_debug_${prefix}.log
echo ${log_file}
CUDA_VISIBLE_DEVICES=0,1 python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --cl_method_name "simple_replay" \
    --replay_size 10 \
    --use_sampled_upstream \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --prefix ${prefix} \
    --save_all_ckpts 1 \
    --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
    --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 



n_gpus=8
n_threads=16
start_gpuid=0 
declare -a prefixes=("nq_dev_0709_simplereplay_rsz10_3e-5_e5")
for prefix in "${prefixes[@]}"
do
    mkdir bug_data/output/${prefix}_offline_eval/
    # for thread in {0..${n_gpus}}
    for (( thread=0; thread<${n_threads}; thread++ ))
    do  
        log_file=bug_data/output/${prefix}_offline_eval/${prefix}_${thread}.log
        gpu=$(($start_gpuid+${thread}%${n_gpus}))
        CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/run_lifelong_finetune.py \
            --prefix ${prefix} \
            --num_threads_eval ${n_threads} --current_thread_id ${thread} --max_timecode 50 \
            --path_to_thread_result bug_data/output/${prefix}_offline_eval/thread_${thread}_of_${n_threads}_result.json \
            --predict_batch_size 40 \
            --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
            --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
        echo ${log_file}
    done 
done