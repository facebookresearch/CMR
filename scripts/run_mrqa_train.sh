task="mrqa_squad"

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
num_epochs=30
output_dir="out/${task}_bart-${modelsize}_1029_upstream_model"

warmup=100
max_input_length=888

# 0617v4

logname="train_bart-${modelsize}_1029"

if [[ -f "data/${task}/${task}_dev.mini.2048.jsonl" ]]
then
    echo "data/${task}/${task}_dev.mini.2048.jsonl exists on your filesystem."
else
    shuf -n 2048 "data/${task}/${task}_dev.jsonl" > "data/${task}/${task}_dev.mini.2048.jsonl"
    echo "data/${task}/${task}_dev.mini.2048.jsonl generated."
fi 

logfile=logs/${task}.${logname}.log
python semanticdebugger/cli_bart.py \
        --do_train \
        --output_dir ${output_dir} \
        --model facebook/bart-${modelsize} \
        --dataset mrqa \
        --train_file data/${task}/${task}_train.jsonl \
        --dev_file data/${task}/${task}_dev.mini.2048.jsonl \
        --test_file data/${task}/${task}_dev.jsonl \
        --learning_rate ${lr} \
        --warmup_steps ${warmup} \
        --train_batch_size ${train_bsz} \
        --predict_batch_size ${pred_bsz} \
        --eval_period 500 \
        --num_train_epochs ${num_epochs} \
        --max_input_length ${max_input_length} \
        --max_output_length 50 \
        --num_beams 3 \
        --append_another_bos  > ${logfile}  2>&1 &

# tail -f logs/${task}.${logname}.log
echo "${logfile}"