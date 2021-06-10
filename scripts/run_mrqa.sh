task="mrqa_naturalquestions"
modelsize="large"
lr=1e-5
train_bsz=8
pred_bsz=32
warmup=100
max_input_length=500

logname="train_bart-${modelsize}"

if [[ -f "data/${task}/${task}_dev.mini.tsv" ]]
then
    echo "data/${task}/${task}_dev.mini.tsv exists on your filesystem."
else
    shuf -n 1000 "data/${task}/${task}_dev.tsv" > "data/${task}/${task}_dev.mini.tsv"
    echo "data/${task}/${task}_dev.mini.tsv generated."
fi



python semanticdebugger/cli_bart.py \
        --do_train \
        --output_dir "out/${task}_bart-${modelsize}" \
        --model facebook/bart-${modelsize} \
        --dataset ${task} \
        --train_file data/${task}/${task}_train.tsv \
        --dev_file data/${task}/${task}_dev.mini.tsv \
        --test_file data/${task}/${task}_dev.tsv \
        --learning_rate ${lr} \
        --warmup_steps ${warmup} \
        --train_batch_size ${train_bsz} \
        --predict_batch_size ${pred_bsz} \
        --eval_period 500 \
        --num_train_epochs 10 \
        --max_input_length ${max_input_length} \
        --max_output_length 50 \
        --num_beams 3 \
        --append_another_bos  > logs/${task}.${logname}.log 2>&1 &

# tail -f logs/${task}.${logname}.log
echo "logs/${task}.${logname}.log"