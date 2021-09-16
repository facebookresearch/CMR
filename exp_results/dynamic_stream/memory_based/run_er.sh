### MbPA++ (w/o local adaptation) = Sparse ER ###
 
num_adapt_epochs=0
memory_store_rate=1.0
prefix=nq_dev_0915_wr_wpara_er
log_file=exp_results/dynamic_stream/memory_based/run_${prefix}.log
mkdir exp_results/dynamic_stream/memory_based/${prefix}_ckpts/
CUDA_VISIBLE_DEVICES=2 python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --max_timecode 100 \
    --cl_method_name "er" \
    --memory_key_encoder "facebook/bart-base" \
    --memory_store_rate ${memory_store_rate} \
    --num_adapt_epochs ${num_adapt_epochs} \
    --replay_size 10 --replay_frequency 1 \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.wr.wpara.json \
    --replay_stream_json_path "" \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
    --save_all_ckpts 0 \
    --memory_path exp_results/dynamic_stream/memory_based/${prefix}_ckpts/memory_dict.pkl \
    --memory_key_cache_path "na" \
    --overtime_ckpt_dir exp_results/dynamic_stream/memory_based/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/memory_based/${prefix}_result.json > ${log_file} 2>&1 & 

echo ${log_file}