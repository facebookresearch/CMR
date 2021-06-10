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


def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--input_file", default="data/mrqa_naturalquestions/mrqa_naturalquestions_dev.tsv", required=False)     
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
    bart_api.inference_api(config_file="scripts/infer_mrqa_bart_base.config", 
                            test_file="data/mrqa_naturalquestions/mrqa_naturalquestions_dev.tsv", 
                            logger=logger)

if __name__=='__main__':
    main()
