
# For the nq_dev 
prefix=mrqa_naturalquestions_dev_bart-base_0617v4_build_bug
CUDA_VISIBLE_DEVICES=0 python semanticdebugger/incre_bench/build_bugpool.py \
    --data_file "data/mrqa_naturalquestions/mrqa_naturalquestions_dev.jsonl" \
    --bug_file "bug_data/mrqa_naturalquestions_dev.bugs.jsonl" \
    --conig_file "scripts/infer_mrqa_bart_base.config" \
    --prefix ${prefix}  > logs/tmp/${prefix}.log 2>&1 &

echo logs/tmp/${prefix}.log

# For the trivia_dev 
prefix=mrqa_triviaqa_dev_bart-base_0617v4_build_bug
CUDA_VISIBLE_DEVICES=1 python semanticdebugger/incre_bench/build_bugpool.py \
    --data_file "data/mrqa_triviaqa/mrqa_triviaqa_dev.jsonl" \
    --bug_file "bug_data/mrqa_triviaqa_dev.bugs.jsonl" \
    --conig_file "scripts/infer_mrqa_bart_base.config" \
    --prefix ${prefix}  > logs/tmp/${prefix}.log 2>&1 &

echo logs/tmp/${prefix}.log


prefix=mrqa_squad_dev_bart-base_0617v4_build_bug
CUDA_VISIBLE_DEVICES=1 python semanticdebugger/incre_bench/build_bugpool.py \
    --data_file "data/mrqa_squad/mrqa_squad_dev.jsonl" \
    --bug_file "bug_data/mrqa_squad_dev.bugs.jsonl" \
    --conig_file "scripts/infer_mrqa_bart_base.config" \
    --prefix ${prefix}  > logs/tmp/${prefix}.log 2>&1 &

echo logs/tmp/${prefix}.log



# prefix=mrqa_naturalquestions_train_bart-base
# CUDA_VISIBLE_DEVICES=1 python semanticdebugger/incre_bench/build_bugpool.py \
#     --data_file "data/mrqa_naturalquestions/mrqa_naturalquestions_train.jsonl" \
#     --bug_file "bug_data/mrqa_naturalquestions_train.bugs.jsonl" \
#     --conig_file "scripts/infer_mrqa_bart_base.config" \
#     --prefix ${prefix}  > logs/${prefix}.log 2>&1 &

