# prefix=mrqa_naturalquestions_dev_bart-base
# CUDA_VISIBLE_DEVICES=0 python semanticdebugger/incre_bench/build_bugpool.py \
#     --data_file "data/mrqa_naturalquestions/mrqa_naturalquestions_dev.tsv" \
#     --bug_file "bug_data/mrqa_naturalquestions_dev.bugs.jsonl" \
#     --conig_file "scripts/infer_mrqa_bart_base.config" \
#     --prefix ${prefix}  > logs/${prefix}.log 2>&1 &

prefix=mrqa_naturalquestions_train_bart-base
CUDA_VISIBLE_DEVICES=1 python semanticdebugger/incre_bench/build_bugpool.py \
    --data_file "data/mrqa_naturalquestions/mrqa_naturalquestions_train.tsv" \
    --bug_file "bug_data/mrqa_naturalquestions_train.bugs.jsonl" \
    --conig_file "scripts/infer_mrqa_bart_base.config" \
    --prefix ${prefix}  > logs/${prefix}.log 2>&1 &

