task="glue_qnli"
# python semanticdebugger/cli_bart.py --do_train \
#         --output_dir out/${task} \
#         --model facebook/bart-base \
#         --dataset ${task} \
#         --train_file data/${task}/${task}_train.tsv \
#         --dev_file data/${task}/${task}_dev.tsv \
#         --learning_rate 3e-5 \
#         --train_batch_size 128 \
#         --predict_batch_size 128 \
#         --eval_period 300 \
#         --num_train_epochs 10 \
#         --max_input_length 64 \
#         --max_output_length 16 \
#         --num_beams 3 \
#         --append_another_bos

python semanticdebugger/cli_bart.py --do_train \
        --output_dir out/${task} \
        --model facebook/bart-large \
        --dataset ${task} \
        --train_file data/${task}/${task}_train.tsv \
        --dev_file data/${task}/${task}_dev.tsv \
        --learning_rate 1e-5 \
        --warmup_steps 600 \
        --train_batch_size 100 \
        --predict_batch_size 128 \
        --eval_period 300 \
        --num_train_epochs 10 \
        --max_input_length 64 \
        --max_output_length 16 \
        --num_beams 3 \
        --append_another_bos