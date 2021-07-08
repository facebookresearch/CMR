# declare -a prefixes=("nq_dev_0701v3_3e-5_e5" "nq_dev_0701v3_1e-5_e3"  "nq_dev_0701v3_3e-5_e3"  "nq_dev_0701v3_1e-5_e5")
# declare -a prefixes=("nq_dev_0701v3_3e-5_e5")
declare -a prefixes=("nq_dev_0706_3e-5_e5" "nq_dev_0706_1e-5_e3" "nq_dev_0706_3e-5_e3" "nq_dev_0706_1e-5_e5")
for prefix in "${prefixes[@]}"
do
    mkdir bug_data/output/${prefix}_offline_eval/
    for thread in {0..7}
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

