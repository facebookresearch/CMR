seed=42
prefix="0924_MixedAllError_T=100_nonecl_seed=${seed}"
log_file=exp_results/dynamic_stream/none/logs/run_${prefix}.log
echo ${log_file}
touch ${log_file}
CUDA_VISIBLE_DEVICES=0 python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --seed $seed \
    --cl_method "none_cl" \
    --num_beams 4 \
    --predict_batch_size 64 \
    --max_timecode 100 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --data_stream_json_path exp_results/data_streams/mrqa.mixed.data_stream.test.json \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa.mixed.upstream_eval.jsonl \
    --replay_stream_json_path "" \
    --save_all_ckpts 0 \
    --overtime_ckpt_dir exp_results/dynamic_stream/none/ckpt_dir/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/none/results/${prefix}_result.json > ${log_file} 2>&1 & 
# tail -f ${log_file}

# --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.wr.wpara.json \
# --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \