################################# For the upstream-train #################################

# n_threads=8
# n_gpus=8
# start_gpuid=0
# declare -a data_names=("mrqa_squad")

# for data_name in "${data_names[@]}"
# do
#     split_name="train"
#     dataset_name="${data_name}_${split_name}"
#     prefix=${dataset_name}_bart-base_0815_inference
#     for (( thread=0; thread<${n_threads}; thread++ ))
#     do 
#         gpu=$(($start_gpuid + $thread % n_gpus))
#         echo $thread, $gpu
#         CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/benchmark_gen/run_bart_infer.py \
#             --data_dist --num_shards $n_threads --local_id ${thread} \
#             --data_file "data/${data_name}/${dataset_name}.jsonl" \
#             --prediction_file "upstream_resources/nli_upstream_preds/tmp/${dataset_name}.predictions.${thread}_of_${n_threads}.json" \
#             --conig_file "scripts/infer_mrqa_bart_base.config" \
#             --prefix ${prefix}  > logs/tmp/${prefix}_${thread}_of_${n_threads}.log 2>&1 &
#         echo logs/tmp/${prefix}_${thread}_of_${n_threads}.log
#     done
#     wait;
# done


# declare -a data_names=("mrqa_squad")
# n_threads=8
# for data_name in "${data_names[@]}"
# do
#     split_name="train"
#     dataset_name="${data_name}_${split_name}"
#     echo $dataset_name
#     python semanticdebugger/benchmark_gen/merge_json_file.py \
#         --input_file_pattern "upstream_resources/nli_upstream_preds/tmp/${dataset_name}.predictions.#_of_${n_threads}.json" \
#         --output_file "upstream_resources/nli_upstream_preds/${dataset_name}.predictions.json" \
#         --range "range(${n_threads})" --mode json
# done 

################################# For the downstream-dev #################################

n_threads=8
n_gpus=8
start_gpuid=0
declare -a data_names=("snli/snli_validation" \
                        "multi_nli/multi_nli_validation_matched" \
                        "multi_nli/multi_nli_validation_mismatched" \
                        "anli/anli_dev" \
                        "scitail/scitail_dev" \
                        )
for data_name in "${data_names[@]}"
do 
    data_name_=$(echo "$data_name" | tr '/' '-')
    prefix=${data_name_}_bart-base_1109_inference
    for (( thread=0; thread<${n_threads}; thread++ ))
    do 
        pred_path="upstream_resources/nli_upstream_preds/tmp/${data_name_}.predictions.${thread}_of_${n_threads}.json" 
        log_path="logs/tmp/${prefix}_${thread}_of_${n_threads}.log" 
        gpu=$(($start_gpuid + $thread % n_gpus))
        echo $thread, $gpu
        CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/benchmark_gen/run_bart_infer.py \
            --data_dist --num_shards $n_threads --local_id ${thread} \
            --data_file "data/${data_name}.jsonl" \
            --prediction_file ${pred_path} \
            --conig_file "scripts/infer_mrqa_bart_base.config" \
            --prefix ${prefix}  > $log_path 2>&1 &
        echo logs/tmp/${prefix}_${thread}_of_${n_threads}.log
    done
    wait;
done

# combine all data

n_threads=8

declare -a data_names=("snli/snli_validation" \
                        "multi_nli/multi_nli_validation_matched" \
                        "multi_nli/multi_nli_validation_mismatched" \
                        "anli/anli_dev" \
                        "scitail/scitail_dev" \
                        )
for data_name in "${data_names[@]}"
do 
    # dataset_name="${data_name}_${split_name}"
    # echo $dataset_name
    data_name_=$(echo "$data_name" | tr '/' '-')
    pred_path="upstream_resources/nli_upstream_preds/tmp/${data_name_}.predictions.#_of_${n_threads}.json" 
    python semanticdebugger/benchmark_gen/merge_json_file.py \
        --input_file_pattern ${pred_path} \
        --output_file "upstream_resources/nli_upstream_preds/${data_name_}.predictions.json" \
        --range "range(${n_threads})" --mode json
done 

# rm upstream_resources/nli_upstream_preds/*.predictions.*_of_8.jsonl