# Semantic Debugger 

## Introduction



## Installation

```bash
# Create a new conda environment (optional)
conda create -n bartqa python=3.6.9
conda activate bartqa
# For building the NLP Few-shot Gym
pip install datasets==1.4.0 py7zr wget
# For reproducing the baseline methods
pip install torch==1.1.0 higher==0.2.1 scikit-learn==0.24.1 scipy==1.4.1 
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
pip install -e .
```


## Cleaning
```bash
python data/data_formatter.py
rm data/*/*-BartTokenized.json
```



## Base Training 

```bash
bash scripts/run_mrqa.sh
```


## Debugging 
```bash
bash scripts/build_bugs_mrqa.sh

python semanticdebugger/incre_bench/sample_bug_stream.py \
    --bug_pool_file "bug_data/mrqa_naturalquestions_dev.bugs.jsonl" \
    --batch_size 20 \
    --num_batches 50 \
    --bug_strema_file "bug_data/mrqa_naturalquestions_dev.static_bug_stream.json"

log_file=logs/tmp/online_debug_0623.log
python semanticdebugger/debug_algs/run_debug.py > ${log_file} 2>&1 &
tail -f ${log_file}
```