

# # For the nq_train
# prefix=mrqa_naturalquestions_train_bart-base_0729_inference
# # CUDA_VISIBLE_DEVICES=0 
# NGPU=7
# GPU_START=1

# for shard_id in $(eval echo "{0..$(( NGPU-1 ))}") 
# do 
#     CUDA_VISIBLE_DEVICES=$((GPU_START+ shard_id)) python semanticdebugger/benchmark_gen/run_bart_infer.py \
#         --data_dist --num_shards $NGPU --local_id ${shard_id} \
#         --data_file "data/mrqa_naturalquestions/mrqa_naturalquestions_train.jsonl" \
#         --prediction_file "bug_data/mrqa_naturalquestions_train.predictions.${shard_id}.jsonl" \
#         --conig_file "scripts/infer_mrqa_bart_base.config" \
#         --prefix ${prefix}  > logs/tmp/${prefix}_${shard_id}.log 2>&1 &
#     echo logs/tmp/${prefix}_${shard_id}.log
# done
# wait;

# python semanticdebugger/benchmark_gen/run_bart_infer.py \
#     --post_process \
#     --num_shards $NGPU \
#     --data_file "data/mrqa_naturalquestions/mrqa_naturalquestions_train.jsonl" \
#     --prediction_file "bug_data/mrqa_naturalquestions_train.predictions.shard_id.jsonl" \
#     --conig_file "scripts/infer_mrqa_bart_base.config" \
#     --prefix ${prefix}



# For the nq_dev
prefix=mrqa_naturalquestions_dev_bart-base_0729_inference
# CUDA_VISIBLE_DEVICES=0 
NGPU=7
GPU_START=1

for shard_id in $(eval echo "{0..$(( NGPU-1 ))}") 
do 
    CUDA_VISIBLE_DEVICES=$((GPU_START+ shard_id)) python semanticdebugger/benchmark_gen/run_bart_infer.py \
        --data_dist --num_shards $NGPU --local_id ${shard_id} \
        --data_file "data/mrqa_naturalquestions/mrqa_naturalquestions_dev.jsonl" \
        --prediction_file "bug_data/mrqa_naturalquestions_dev.predictions.${shard_id}.jsonl" \
        --conig_file "scripts/infer_mrqa_bart_base.config" \
        --prefix ${prefix}  > logs/tmp/${prefix}_${shard_id}.log 2>&1 &
    echo logs/tmp/${prefix}_${shard_id}.log
done
wait;

# python semanticdebugger/benchmark_gen/run_bart_infer.py \
#     --post_process \
#     --num_shards $NGPU \
#     --data_file "data/mrqa_naturalquestions/mrqa_naturalquestions_dev.jsonl" \
#     --prediction_file "bug_data/mrqa_naturalquestions_dev.predictions.shard_id.jsonl" \
#     --conig_file "scripts/infer_mrqa_bart_base.config" \
#     --prefix ${prefix}



# tail -f logs/tmp/${prefix}.log


# # For the nq_dev 
# prefix=mrqa_naturalquestions_dev_bart-base_0617v4_build_bug
# CUDA_VISIBLE_DEVICES=0 python semanticdebugger/benchmark_gen/build_bugpool.py \
#     --data_file "data/mrqa_naturalquestions/mrqa_naturalquestions_dev.jsonl" \
#     --bug_file "bug_data/mrqa_naturalquestions_dev.bugs.jsonl" \
#     --conig_file "scripts/infer_mrqa_bart_base.config" \
#     --prefix ${prefix}  > logs/tmp/${prefix}.log 2>&1 &

# echo logs/tmp/${prefix}.log

# # For the trivia_dev 
# prefix=mrqa_triviaqa_dev_bart-base_0617v4_build_bug
# CUDA_VISIBLE_DEVICES=1 python semanticdebugger/benchmark_gen/build_bugpool.py \
#     --data_file "data/mrqa_triviaqa/mrqa_triviaqa_dev.jsonl" \
#     --bug_file "bug_data/mrqa_triviaqa_dev.bugs.jsonl" \
#     --conig_file "scripts/infer_mrqa_bart_base.config" \
#     --prefix ${prefix}  > logs/tmp/${prefix}.log 2>&1 &

# echo logs/tmp/${prefix}.log


# prefix=mrqa_squad_dev_bart-base_0617v4_build_bug
# CUDA_VISIBLE_DEVICES=1 python semanticdebugger/benchmark_gen/build_bugpool.py \
#     --data_file "data/mrqa_squad/mrqa_squad_dev.jsonl" \
#     --bug_file "bug_data/mrqa_squad_dev.bugs.jsonl" \
#     --conig_file "scripts/infer_mrqa_bart_base.config" \
#     --prefix ${prefix}  > logs/tmp/${prefix}.log 2>&1 &

# echo logs/tmp/${prefix}.log



# prefix=mrqa_naturalquestions_train_bart-base
# CUDA_VISIBLE_DEVICES=1 python semanticdebugger/benchmark_gen/build_bugpool.py \
#     --data_file "data/mrqa_naturalquestions/mrqa_naturalquestions_train.jsonl" \
#     --bug_file "bug_data/mrqa_naturalquestions_train.bugs.jsonl" \
#     --conig_file "scripts/infer_mrqa_bart_base.config" \
#     --prefix ${prefix}  > logs/${prefix}.log 2>&1 &

