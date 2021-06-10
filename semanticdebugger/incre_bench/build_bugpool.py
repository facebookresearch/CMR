from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random
import numpy as np
import torch
from semanticdebugger.incre_bench import bart_api
from semanticdebugger.task_manager.eval_metrics import evaluate_func

def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--input_file", default="data/mrqa_naturalquestions/mrqa_naturalquestions_dev.100.tsv", required=False)     
    parser.add_argument("--output_file", default="bug_data/mrqa_naturalquestions_dev.bugs.tsv", required=False)
    parser.add_argument("--model_conigfile", default="scripts/infer_mrqa_bart_base.config", required=False)
    parser.add_argument("--prefix", default="", required=False)
    ## API for Evaluation
    
    ## Sampling
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    log_filename = "logs/{}_build_bugpool_log.txt".format("")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(log_filename)),
                                logging.StreamHandler()])
    logger = logging.getLogger(__name__)


    # get the truth data 
    truth_data = []
    with open(args.input_file) as fin:
        lines = fin.readlines()
    # train_examples = []
    for line in lines:
        d = line.strip().split("\t")
        truth_data.append((d[0], d[1:]))

    # get the predictions of a model via its API and config file.

    predictions = bart_api.inference_api(
                            config_file=args.model_conigfile, 
                            test_file=args.input_file, 
                            logger=logger)
    
    # get evaluation results.
    metric = "EM|QA-F1"
    results, results_all = evaluate_func(predictions, truth_data, metric, return_all=True)
    print(results)
    print(results_all["EM"])
    print(results_all["QA-F1"])
    

    

if __name__=='__main__':
    main()
