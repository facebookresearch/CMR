# On Continual Model Refinement in Out-of-Distribution Data Streams


### **_Quick links:_**  [**[Paper]**](https://yuchenlin.xyz/files/cmr.pdf)   [**[Documentation]**](https://cmr-nlp.github.io/)

---
This is the repository of the paper, [_**On Continual Model Refinement in Out-of-Distribution Data Streams**_](https://yuchenlin.xyz/files/cmr.pdf), by [_Bill Yuchen Lin_](https://yuchenlin.xyz/), [Sida Wang](http://www.sidaw.xyz/), [Xi Victoria Lin](http://victorialin.net/), [Robin Jia](https://robinjia.github.io/), [Lin Xiao](https://linxiaolx.github.io/), [Xiang Ren](http://www-bcf.usc.edu/~xiangren/), and [Scott Yih](https://scottyih.org/), published in Proc. of [*ACL 2022*](https://www.2022.aclweb.org/). 

---

## Abstract 
Real-world natural language processing (NLP) models need to be continually updated to fix the prediction errors in out-of-distribution (OOD) data streams while overcoming catastrophic forgetting. However, existing continual learning (CL) problem setups cannot cover such a realistic and complex scenario. In response to this, we propose a new CL problem formulation dubbed continual model refinement (CMR). Compared to prior CL settings, CMR is more practical and introduces unique challenges (boundary-agnostic and non-stationary distribution shift, diverse mixtures of multiple OOD data clusters, error-centric streams, etc.). We extend several existing CL approaches to the CMR setting and evaluate them extensively. For benchmarking and analysis, we propose a general sampling algorithm to obtain dynamic OOD data streams with controllable nonstationarity, as well as a suite of metrics measuring various aspects of online performance. Our experiments and detailed analysis reveal the promise and challenges of the CMR problem, supporting that studying CMR in dynamic OOD streams can benefit the longevity of deployed NLP models in production.

## Installation

```bash
# Create a new conda environment (optional)
conda create -n cmr python=3.6.9
conda activate cmr
pip install datasets==1.4.0 py7zr wget
pip install torch==1.4.0 higher==0.2.1 scikit-learn==0.24.1 scipy==1.4.1 
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
pip install -e .

conda install -n cmr -c pytorch faiss-gpu
```

## Data preprocessing

Here we use the MRQA datasets as an example to show how datasets should be processed.

1. download the data files 
```bash
cd data/mrqa/
bash download.sh
```

2. preprocess the datasets
```bash
cd ~/CMR/ # go to the root folder of CMR project, say ~/CMR/
python data/data_formatter.py
```

After thest two steps, you should see a few `mrqa_*` folders under the `data` folder, where each is for a particular data cluster.

## Upstream learning

Now we use a particular data cluster, say `mrqa_squad`, to train an upstream model for the interested task, e.g., extractive question answering. 
Here, we use `bart-base` as our base LM and train the upstream model in a text-to-text manner. Please find more details for the arguments in our documentation.

```bash 
bash scripts/run_mrqa_train.sh  # under the CMR folder
```

After the upstream training, we should have a model checkpoint in `out/mrqa_squad_bart-base_upstream_model`.

## CMR Setup

1. Generate upstream model predictions.

We first use the upstream model to infer examples from other data clusters.

```bash
mkdir -p upstream_resources/qa_upstream_preds/tmp/ # under the CMR folder
bash scripts/run_mrqa_infer.sh
```

The first part of this script is to test the upstream model on the upstream training data, and the second part is to test the upstream model on other data clusters' dev data. 

2. Generate data streams.

Now we generate the data streams and evaluation sets that we need for our experiments. The default configurations that are used here can be found in the code file. 
We will show how to change specific configurations for OOD stream sampling in our project documentation.
```bash
mkdir -p experiments/eval_data/qa/
python cmr/benchmark_gen/sample_submission_streams.py --task_name QA
```

The generated data streams can be visualized by running `cmr/benchmark_gen/visualize_streams.py`.

## Experiments 

There are multiple scripts for running experiments with different CMR methods in `experiments/run_scripts`. 
We leave their documentation on our project website.
Here we take the `run_simplecl.sh` and `slurm_simple.sh` as examples for running the Continual-Finetuning baseline method shown in our paper. For running other CMR methods, the filename should be self-explanatory and we will show the detailed instructions on our website.

For running a particular setup of CMR experiments, 
we can execute the below command:
```bash
bash experiments/run_scripts/run_simplecl.sh ${seed} ${lr} ${ep} ${l2w} ${ns_config} ${task} val/test ${stream_id}
```

### Wrapper

Here, the arguments, such as `seed`, `lr`, are usually specified in a wrapper script, which in this example is the `slurm_simple.sh`.
The wrapper script has two major modes: validation (`val`) and testing (`test`). 
The validation mode is to explore the best hps and the testing mode is to evaluate the final hps of a CMR method with multiple rounds of experiments (via setting different seeds and using different streams).

### Job scheduling 
Note that we include a slurm-related command for those who use [slurm] for scheduling jobs on their servers. If you don't use slurm, you could remove the following part (i.e., `tmux ...`) or replace it with your own job-scheduling method for running the job `[xxx]`:

`tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=120 --cpus-per-task 4 --pty [xxx]"`

### CMR methods 

For more descriptions of the CMR methods that we support in this codebase, please refer to our project website. 
All CMR methods are based on the `OnlineDebuggingMethod` class in `cmr/debug_algs/commons.py` and they are written in different files named `cmr/debug_algs/cl_*.py`.


## Evaluation 

The online evaluation is part of the code for CMR methods and the results are saved into json-format result files, so we do not need to run a separate script for evaluation.
We provide a script to generate a csv-format file to report the performance of multiple CMR methods. Please check out `experiments/report_results.py`.


## Citation 

```bibtex
@inproceedings{lin-etal-2022-cmr,
    title = "On Continual Model Refinement in Out-of-Distribution Data Streams",
    author = "Lin, Bill Yuchen and Wang, Sida and Lin, Xi Victoria and Jia, Robin and Xiao, Lin and Ren, Xiang and Yih, Wen-tau",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022)",
    year = "2022"
}
```
 

## Contact 

Please email yuchen.lin@usc.edu if you have any questions. 

## License

See the [LICENSE](LICENSE.md) file for more details.
