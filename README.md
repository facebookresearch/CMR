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
```



## Base Training 

```bash
bash src/scripts/run_triviaqa.sh > train_trivaqa.log 2>&1 &
tail -f train_trivaqa.log

bash src/scripts/run_qnli.sh > train_qnli.log 2>&1 &
tail -f train_qnli.log
```
