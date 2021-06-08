task="mrqa_naturalquestions"
logname="train_bart-base"

if [[ -f "data/${task}/${task}_dev.mini.tsv" ]]
then
    echo "data/${task}/${task}_dev.mini.tsv exists on your filesystem."
else
    shuf -n 1000 "data/${task}/${task}_dev.tsv" > "data/${task}/${task}_dev.mini.tsv"
fi



python src/cli_base.py \
        --do_train \
        --output_dir out/${task} \
        --model facebook/bart-base \
        --dataset ${task} \
        --train_file data/${task}/${task}_train.tsv \
        --dev_file data/${task}/${task}_dev.mini.tsv \
        --test_file data/${task}/${task}_dev.tsv \
        --learning_rate 3e-5 \
        --warmup_steps 100 \
        --train_batch_size 32 \
        --predict_batch_size 64 \
        --eval_period 500 \
        --num_train_epochs 10 \
        --max_input_length 888 \
        --max_output_length 50 \
        --num_beams 3 \
        --append_another_bos  > logs/${task}.${logname}.log 2>&1 &

# tail -f logs/${task}.${logname}.log
echo "logs/${task}.${logname}.log"