task="mrqa_squad"
CUDA_VISIBLE_DEVICES=1 python src/cli_base.py \
        --do_train \
        --output_dir out/${task} \
        --model facebook/bart-large \
        --dataset ${task} \
        --train_file data/${task}/${task}_train.tsv \
        --dev_file data/${task}/${task}_dev.tsv \
        --test_file data/${task}/${task}_dev.tsv \
        --learning_rate 1e-5 \
        --warmup_steps 100 \
        --train_batch_size 1 \
        --predict_batch_size 1 \
        --eval_period 300 \
        --num_train_epochs 10 \
        --max_input_length 888 \
        --max_output_length 50 \
        --num_beams 3 \
        --append_another_bos