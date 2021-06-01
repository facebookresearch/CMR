task="kilt_triviaqa"
python src/cli_base.py --do_train \
        --output_dir out/${task} \
        --model facebook/bart-base \
        --dataset ${task} \
        --train_file data/${task}/${task}_train.tsv \
        --dev_file data/${task}/${task}_dev.tsv \
        --train_batch_size 16 \
        --predict_batch_size 32 \
        --eval_period 100 \
        --append_another_bos