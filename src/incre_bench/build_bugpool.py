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

from run_bart import run

def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--input_file", default="path to evaluate and build the pool of bugs", required=True)     
    parser.add_argument("--output_file", default="path to save the bugs", required=True)
    parser.add_argument("--model_conigfile", default="path to load the model parameters", required=True)
    ## API for Evaluation
    
    ## Sampling
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

if __name__=='__main__':
    main()
