# prefix=nq_dev_0701_v2
# mkdir bug_data/output/${prefix}_offline_eval/
# for thread in {0..15}
# do  
#     log_file=bug_data/output/${prefix}_offline_eval/${prefix}_${thread}.log
#     gpu=$(( thread / 2))
#     CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/run_lifelong_finetune.py \
#         --prefix ${prefix} \
#         --num_threads_eval 16 --current_thread_id ${thread} --max_timecode 50 \
#         --path_to_thread_result bug_data/output/${prefix}_offline_eval/thread_${thread}_result.json \
#         --predict_batch_size 88 \
#         --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
#         --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
#     echo ${log_file}
# done


# prefix=nq_dev_0701v3
# mkdir bug_data/output/${prefix}_offline_eval/
# for thread in {0..8}
# do  
#     log_file=bug_data/output/${prefix}_offline_eval/${prefix}_${thread}.log
#     gpu=$thread
#     CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/run_lifelong_finetune.py \
#         --prefix ${prefix} \
#         --num_threads_eval 8 --current_thread_id ${thread} --max_timecode 50 \
#         --path_to_thread_result bug_data/output/${prefix}_offline_eval/thread_${thread}_result.json \
#         --predict_batch_size 88 \
#         --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
#         --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
#     echo ${log_file}
# done


declare -a prefixes=("nq_dev_0706_3e-5_e5")
for prefix in "${prefixes[@]}"
do
    mkdir bug_data/output/${prefix}_offline_eval/
    for thread in {0..8}
    do  
        log_file=bug_data/output/${prefix}_offline_eval/${prefix}_${thread}.log
        gpu=$thread
        CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/run_lifelong_finetune.py \
            --prefix ${prefix} \
            --num_threads_eval 8 --current_thread_id ${thread} --max_timecode 50 \
            --path_to_thread_result bug_data/output/${prefix}_offline_eval/thread_${thread}_result.json \
            --predict_batch_size 80 \
            --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
            --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
        echo ${log_file}
    done
    wait;
done