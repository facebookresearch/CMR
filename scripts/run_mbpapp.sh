# prefix=nq_dev_0716_mbpapp_rsz32_rf30_3e-5_e5
# log_file=logs/tmp/online_debug_${prefix}.log
# echo ${log_file}
# CUDA_VISIBLE_DEVICES=0,1 python semanticdebugger/debug_algs/run_lifelong_finetune.py \
#     --cl_method_name "mbpa++" \
#     --memory_key_encoder "facebook/bart-base" \
#     --memory_store_rate 1.0 \
#     --replay_size 32 \
#     --replay_frequency 30 \
#     --use_sampled_upstream \
#     --learning_rate 3e-5 --num_train_epochs 5 \
#     --prefix ${prefix} \
#     --save_all_ckpts 1 \
#     --memory_path bug_data/output/${prefix}_ckpts/memory_dict.pkl \
#     --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
#     --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 


#### MbPA++ ###

# num_adapt_epochs=1
# n_gpus=8
# n_threads=16
# start_gpuid=0 
# declare -a prefixes=("nq_dev_0716_mbpapp_rsz32_rf30_3e-5_e5")
# for prefix in "${prefixes[@]}"
# do
#     mkdir bug_data/output/${prefix}_offline_eval/
#     # for thread in {0..${n_gpus}}
#     for (( thread=0; thread<${n_threads}; thread++ ))
#     do  
#         log_file=bug_data/output/${prefix}_offline_eval/${prefix}_${thread}_of_${n_threads}.log
#         gpu=$(($start_gpuid+${thread}%${n_gpus}))
#         CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/run_lifelong_finetune.py \
#             --num_threads_eval ${n_threads} --current_thread_id ${thread} --max_timecode 50 \
#             --cl_method_name "mbpa++" \
#             --memory_key_encoder "facebook/bart-base" \
#             --memory_path bug_data/output/${prefix}_ckpts/memory_dict.pkl \
#             --replay_size 8 \
#             --num_adapt_epochs ${num_adapt_epochs} \
#             --prefix ${prefix} \
#             --path_to_thread_result bug_data/output/${prefix}_offline_eval/thread_${thread}_of_${n_threads}_result.json \
#             --train_batch_size 6 \
#             --predict_batch_size 8 \
#             --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
#             --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
#         echo ${log_file}
#     done 
# done

#### MbPA++ (w/o sparse ER) = MbPA ###

num_adapt_epochs=1
n_gpus=8
n_threads=16
start_gpuid=0 

base_prefix="nq_dev_0716_mbpapp_rsz32_rf30_3e-5_e5"
in_prefix="nq_dev_0706_3e-5_e5"
out_prefix="nq_dev_0716_mbpa_3e-5_e5"
memory_path=bug_data/output/${base_prefix}_ckpts/memory_dict.pkl
mkdir bug_data/output/${out_prefix}_offline_eval/
# for thread in {0..${n_gpus}}
for (( thread=0; thread<${n_threads}; thread++ ))
do  
    log_file=bug_data/output/${out_prefix}_offline_eval/${out_prefix}_${thread}_of_${n_threads}.log
    gpu=$(($start_gpuid+${thread}%${n_gpus}))
    CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/run_lifelong_finetune.py \
        --num_threads_eval ${n_threads} --current_thread_id ${thread} --max_timecode 50 \
        --cl_method_name "mbpa++" \
        --memory_key_encoder "facebook/bart-base" \
        --memory_path ${memory_path} \
        --replay_size 8 \
        --num_adapt_epochs ${num_adapt_epochs} \
        --prefix ${out_prefix} \
        --path_to_thread_result bug_data/output/${out_prefix}_offline_eval/thread_${thread}_of_${n_threads}_result.json \
        --train_batch_size 6 \
        --predict_batch_size 8 \
        --overtime_ckpt_dir bug_data/output/${in_prefix}_ckpts/ \
        --result_file bug_data/output/${out_prefix}_result.json > ${log_file} 2>&1 & 
    echo ${log_file} 
done 

#### MbPA++ (w/o local adaptation) = Sparse ER ###
 
# num_adapt_epochs=0
# n_gpus=8
# n_threads=16
# start_gpuid=0 
# declare -a prefixes=("nq_dev_0716_mbpapp_rsz32_rf30_3e-5_e5")
# for prefix in "${prefixes[@]}"
# do
#     mkdir bug_data/output/${prefix}_offline_eval/
#     mkdir bug_data/output/${prefix}_woadapt_offline_eval/
#     # for thread in {0..${n_gpus}}
#     for (( thread=0; thread<${n_threads}; thread++ ))
#     do  
#         log_file=bug_data/output/${prefix}_offline_eval/${prefix}_woadapt_${thread}_of_${n_threads}.log
#         gpu=$(($start_gpuid+${thread}%${n_gpus}))
#         CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/run_lifelong_finetune.py \
#             --num_threads_eval ${n_threads} --current_thread_id ${thread} --max_timecode 50 \
#             --cl_method_name "mbpa++" \
#             --replay_size 8 \
#             --num_adapt_epochs ${num_adapt_epochs} \
#             --prefix ${prefix} \
#             --path_to_thread_result bug_data/output/${prefix}_woadapt_offline_eval/thread_${thread}_of_${n_threads}_result.json \
#             --train_batch_size 1 \
#             --predict_batch_size 16 \
#             --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
#             --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
#         echo ${log_file}
#     done 
# done



# Debugging the MBPA++ local adaptation

# num_adapt_epochs=1
# n_gpus=8
# n_threads=100
# start_gpuid=0 
# declare -a prefixes=("nq_dev_0716_mbpapp_rsz32_rf30_3e-5_e5")
# for prefix in "${prefixes[@]}"
# do
#     mkdir bug_data/output/${prefix}_offline_eval/
#     # for (( thread=0; thread<${n_threads}; thread++ ))
#     # do  
#         thread=0
#         log_file=bug_data/output/${prefix}_offline_eval/${prefix}_${thread}_of_${n_threads}.log
#         gpu=$(($start_gpuid+${thread}%${n_gpus}))
#         CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/run_lifelong_finetune.py \
#             --num_threads_eval ${n_threads} --current_thread_id ${thread} --max_timecode 50 \
#             --cl_method_name "mbpa++" \
#             --memory_key_encoder "facebook/bart-base" \
#             --memory_path bug_data/output/${prefix}_ckpts/memory_dict.pkl \
#             --replay_size 8 \
#             --num_adapt_epochs ${num_adapt_epochs} \
#             --prefix ${prefix} \
#             --path_to_thread_result bug_data/output/${prefix}_offline_eval/thread_${thread}_of_${n_threads}_result.json \
#             --train_batch_size 6 \
#             --predict_batch_size 8 \
#             --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
#             --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
#         echo ${log_file}
#     # done 
# done


# num_adapt_epochs=0
# n_gpus=8
# n_threads=100
# start_gpuid=1 
# declare -a prefixes=("nq_dev_0716_mbpapp_rsz32_rf30_3e-5_e5")
# for prefix in "${prefixes[@]}"
# do    
#     mkdir bug_data/output/${prefix}_woadapt_offline_eval/
#     # for (( thread=0; thread<${n_threads}; thread++ ))
#     # do  
#         thread=0
#         log_file=bug_data/output/${prefix}_offline_eval/${prefix}_woadapt_${thread}_of_${n_threads}.log
#         gpu=$(($start_gpuid+${thread}%${n_gpus}))
#         CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/debug_algs/run_lifelong_finetune.py \
#             --num_threads_eval ${n_threads} --current_thread_id ${thread} --max_timecode 50 \
#             --cl_method_name "mbpa++" \
#             --memory_key_encoder "facebook/bart-base" \
#             --memory_path bug_data/output/${prefix}_ckpts/memory_dict.pkl \
#             --replay_size 8 \
#             --num_adapt_epochs ${num_adapt_epochs} \
#             --prefix ${prefix} \
#             --path_to_thread_result bug_data/output/${prefix}_woadapt_offline_eval/thread_${thread}_of_${n_threads}_result.json \
#             --train_batch_size 6 \
#             --predict_batch_size 20 \
#             --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ \
#             --result_file bug_data/output/${prefix}_result.json > ${log_file} 2>&1 & 
#         echo ${log_file}
#     # done 
# done