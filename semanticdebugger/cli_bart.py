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

from semanticdebugger.models.run_bart import run


def get_parser():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--train_file", default="data", required=False)
    parser.add_argument("--dev_file", default="data", required=False)
    parser.add_argument("--test_file", default="data", required=False)
    parser.add_argument("--dataset", default="None", required=False)
    parser.add_argument("--model", default="facebook/bart-base", required=False)

    parser.add_argument("--output_dir", default=None, type=str, required=False)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--predict_checkpoint", type=str,
                        default="best-model.pt")

    # Model parameters
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument("--freeze_embeds", action='store_true', default=False)

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=128)
    parser.add_argument('--max_output_length', type=int, default=32)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos",
                        action='store_true', default=False)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--warmup_proportion", default=0.01, type=float,
    #                     help="Weight decay if we apply some.")    # Not used
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=300, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--total_steps", default=100000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10)

    # Other parameters
    parser.add_argument("--quiet", action='store_true',
                        help="If true, tqdm will not show progress bar")
    parser.add_argument('--eval_period', type=int, default=100,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    return parser


def main():
    args = get_parser().parse_args()
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Start writing logs

    log_filename = "{}log.txt".format("train_" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_dir` must be specified.")
        if not args.dev_file:
            raise ValueError(
                "If `do_train` is True, then `predict_dir` must be specified.")

    if args.do_predict:
        if not args.test_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_dir` must be specified.")

    logger.info("Using {} gpus".format(args.n_gpu))
    run(args, logger)


if __name__ == '__main__':
    main()
