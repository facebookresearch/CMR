### MbPA++ (w/o local adaptation) = Sparse ER ###

candidate_size=$1

gpu=0 
declare -a seeds=("42")
#  "2021" "0212" "1213"

for seed in "${seeds[@]}"
do
num_adapt_epochs=0
memory_store_rate=1.0
# prefix="nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=${candidate_size}_seed=${seed}"
prefix="nq_dev_0920debug_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=${candidate_size}_seed=${seed}"
log_file=exp_results/dynamic_stream/memory_based/logs/run_${prefix}.log
tmp_code_copy=exp_results/dynamic_stream/memory_based/logs/${prefix}.cl_mbcl_alg.py
mkdir exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/
cp semanticdebugger/debug_algs/cl_mbcl_alg.py $tmp_code_copy

echo ${log_file}

CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --seed $seed \
    --max_timecode 100 \
    --cl_method_name "mir" \
    --memory_key_encoder "facebook/bart-base" \
    --memory_store_rate ${memory_store_rate} \
    --num_adapt_epochs ${num_adapt_epochs} \
    --replay_candidate_size ${candidate_size} \
    --use_sampled_upstream --use_replay_mix \
    --replay_size 16 --replay_frequency 3 \
    --learning_rate 3e-5 --num_train_epochs 5 \
    --prefix ${prefix} \
    --stream_mode dynamic \
    --data_stream_json_path exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.wr.wpara.json \
    --replay_stream_json_path "" \
    --pass_pool_jsonl_path exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
    --save_all_ckpts 0 \
    --memory_path exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/memory_dict.pkl \
    --memory_key_cache_path "na" \
    --overtime_ckpt_dir exp_results/dynamic_stream/memory_based/ckpt_dir/${prefix}_ckpts/ \
    --result_file exp_results/dynamic_stream/memory_based/results/${prefix}_result.json > ${log_file} 2>&1 &
gpu=$((gpu+1))
done 