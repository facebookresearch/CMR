task="mrqa_naturalquestions"


# modelsize="large"
# lr=1e-5
# train_bsz=8
# pred_bsz=32

# modelsize="base"
# lr=5e-5
# train_bsz=32
# pred_bsz=64

modelsize="base"
lr=5e-5
train_bsz=64
pred_bsz=64

output_dir="out/${task}_bart-${modelsize}_1011"

warmup=100
max_input_length=888

# 0617v4

logname="train_bart-${modelsize}_1011"

if [[ -f "data/${task}/${task}_dev.mini.jsonl" ]]
then
    echo "data/${task}/${task}_dev.mini.jsonl exists on your filesystem."
else
    shuf -n 1000 "data/${task}/${task}_dev.jsonl" > "data/${task}/${task}_dev.mini.jsonl"
    echo "data/${task}/${task}_dev.mini.jsonl generated."
fi



python semanticdebugger/cli_bart.py \
        --do_train \
        --output_dir ${output_dir} \
        --model facebook/bart-${modelsize} \
        --dataset ${task} \
        --train_file data/${task}/${task}_train.jsonl \
        --dev_file data/${task}/${task}_dev.mini.jsonl \
        --test_file data/${task}/${task}_dev.jsonl \
        --learning_rate ${lr} \
        --warmup_steps ${warmup} \
        --train_batch_size ${train_bsz} \
        --predict_batch_size ${pred_bsz} \
        --eval_period 500 \
        --num_train_epochs 5 \
        --max_input_length ${max_input_length} \
        --max_output_length 50 \
        --num_beams 3 \
        --append_another_bos  > logs/tmp/${task}.${logname}.log 2>&1 &

# tail -f logs/${task}.${logname}.log
echo "logs/tmp/${task}.${logname}.log"