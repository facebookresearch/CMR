### MIR ###

max_timecode=$1
replay_size=$2
replay_freq=$3
candidate_size=$4
abalation_args=$5

gpu=0 
declare -a seeds=("42" "0212" "1213")

for seed in "${seeds[@]}"
do
memory_store_rate=1.0 
prefix="0923_MixedAllErrors_T=${max_timecode}_mir_M=I_rs=${replay_size}_rq=${replay_freq}_candidate=${candidate_size}_mode=${abalation_args}_seed=${seed}"
log_file=exp_results/dynamic_stream/memory_based/logs/run_${prefix}.log
# rm exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/memory_dict.pkl
rm -rf exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/
mkdir exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/
tmp_script_copy=exp_results/dynamic_stream/memory_based/logs/${prefix}.run_mir.sh
tmp_code_copy=exp_results/dynamic_stream/memory_based/logs/${prefix}.cl_mbcl_alg.py
cp semanticdebugger/debug_algs/cl_mbcl_alg.py $tmp_code_copy
cp exp_results/dynamic_stream/memory_based/run_mir.sh $tmp_script_copy

echo ${log_file}

# --mir_debug_reverse \
# --mir_debug_largestloss \
# M=U+I --use_sampled_upstream \


CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --mir_abalation_args ${abalation_args} \
    --local_adapt_lr 3e-5 --num_adapt_epochs 3 \
    --seed $seed \
    --max_timecode ${max_timecode} \
    --cl_method_name "mir" \
    --memory_key_encoder "facebook/bart-base" \
    --memory_store_rate ${memory_store_rate} \
    --replay_candidate_size ${candidate_size} \
    --use_replay_mix \
    --replay_size ${replay_size} --replay_frequency ${replay_freq} \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --data_stream_json_path exp_results/data_streams/mrqa.mixed.data_stream.test.json \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa.mixed.hidden_passes.jsonl \
    --replay_stream_json_path "" \
    --save_all_ckpts 1 \
    --memory_path exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/memory_dict.pkl \
    --memory_key_cache_path "na" \
    --overtime_ckpt_dir exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/memory_based/results/${prefix}_result.json > ${log_file} 2>&1 &
gpu=$((gpu+1))
done 