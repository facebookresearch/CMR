# For the nq_train

n_threads=8
n_gpus=8
start_gpuid=0
declare -a data_names=("mrqa_naturalquestions") 

for data_name in "${data_names[@]}"
do
    split_name="train"
    dataset_name="${data_name}_${split_name}"
    prefix=${dataset_name}_bart-base_1011_inference_meta
    for (( thread=0; thread<${n_threads}; thread++ ))
    do 
        gpu=$(($start_gpuid + $thread % n_gpus))
        echo $thread, $gpu
        CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/benchmark_gen/run_bart_infer.py \
            --data_dist --num_shards $n_threads --local_id ${thread} \
            --data_file "data/${data_name}/${dataset_name}.jsonl" \
            --prediction_file "bug_data/1011_${dataset_name}.predictions.${thread}_of_${n_threads}.json" \
            --conig_file "scripts/infer_mrqa_nq_meta_bart_base.config" \
            --prefix ${prefix}  > logs/tmp/${prefix}_${thread}_of_${n_threads}.log 2>&1 &
        echo logs/tmp/${prefix}_${thread}_of_${n_threads}.log
    done
    wait;
done



declare -a data_names=("mrqa_naturalquestions")
n_threads=8
for data_name in "${data_names[@]}"
do
    split_name="train"
    dataset_name="${data_name}_${split_name}"
    echo $dataset_name
    python semanticdebugger/benchmark_gen/merge_json_file.py \
        --input_file_pattern "bug_data/1011_${dataset_name}.predictions.#_of_${n_threads}.json" \
        --output_file "bug_data/1011_${dataset_name}.meta.predictions.json" \
        --range "range(${n_threads})" --mode json
done 

rm bug_data/*.predictions.*_of_8.jsonl