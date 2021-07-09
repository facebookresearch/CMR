from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
from semanticdebugger.models.utils import set_seeds
import sys

import numpy as np
import torch
from semanticdebugger.incre_bench import bart_api
from semanticdebugger.task_manager.eval_metrics import (evaluate_func,
                                                        normalize_answer)


def generate_bugs(predictions, truth_data, results_all):
    assert len(predictions) == len(truth_data) == len(
        results_all["EM"]) == len(results_all["QA-F1"])
    bug_lines = []
    pass_lines = []
    for p, t, em, f1 in zip(predictions, truth_data, results_all["EM"], results_all["QA-F1"]):
        item = dict()
        item["input"] = t[0]
        item["truth"] = t[1]
        item["id"] =  t[2]
        item["mistake"] = p.strip()
        item["score"] = {"EM": int(em == True), "QA-F1": float(f1)}
        if em == False and f1 < 0.5:  # decide later about the threshold of f1 score
            bug_lines.append(json.dumps(item))
        else:
            pass_lines.append(json.dumps(item))
    return bug_lines, pass_lines


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument(
        "--data_file", default="data/mrqa_naturalquestions/mrqa_naturalquestions_dev.100.jsonl", required=False)
    parser.add_argument(
        "--bug_file", default="bug_data/mrqa_naturalquestions_dev.bugs.jsonl", required=False)
    parser.add_argument(
        "--conig_file", default="scripts/infer_mrqa_bart_base.config", required=False)
    parser.add_argument("--prefix", default="", required=False)
    # API for Evaluation

    parser.add_argument("--metric", default="EM|QA-F1", required=False)

    # Sampling
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    set_seeds(args.seed)

    log_filename = "logs/build_bugpool_log_{}.txt".format(args.prefix)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(log_filename)),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    # get the truth data
    truth_data = []
    with open(args.data_file) as fin:
        lines = fin.readlines()
    # train_examples = []
    for line in lines:
        # d = line.strip().split("\t")
        # truth_data.append((d[0], d[1:]))
        d = json.loads(line)
        truth_data.append((d["input"], d["output"], d["id"]))
    # get the predictions of a model via its API and config file.

    predictions = bart_api.inference_api(
        config_file=args.conig_file,
        test_file=args.data_file,
        logger=logger)
    with open(args.bug_file.replace("bugs.jsonl", "predictions.json"), "w") as f:
        json.dump(predictions, f)

    # get evaluation results.
    results, results_all = evaluate_func(
        predictions, truth_data, args.metric, return_all=True)
    logging.info("Evaluation results " + str(results))
    bug_lines, pass_lines = generate_bugs(predictions, truth_data, results_all)
    logging.info("{} example are passed. Found {} bugs ".format(
        len(pass_lines), len(bug_lines)))

    # save the bugs
    with open(args.bug_file, "w") as f:
        f.write("\n".join(bug_lines))

    # save the passes
    with open(args.bug_file.replace("bugs", "pass"), "w") as f:
        f.write("\n".join(pass_lines))


if __name__ == '__main__':
    main()
