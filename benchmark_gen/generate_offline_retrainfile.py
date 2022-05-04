# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import argparse
from types import new_class
import random 

parser = argparse.ArgumentParser()

parser.add_argument(
        "--upstream_file",
        default="data/mrqa_squad/mrqa_squad_train.jsonl", type=str)
parser.add_argument(
        "--submission_file",
        default="experiments/eval_data/qa/submission_stream.T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8.json", type=str)
parser.add_argument(
        "--mixed_offline_file",
        default="experiments/eval_data/qa/offline_retrain.T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8.jsonl", type=str)
parser.add_argument(
        "--heldout_eval_file",
        default="experiments/eval_data/qa/heldout_eval.jsonl", type=str)
        
parser.add_argument("--ratio", default=-1, type=float)

args = parser.parse_args()

with open(args.submission_file, "r") as f :
    data_stream = json.load(f)     

all_init_errors = []
for data_batch in data_stream: 
    for item in data_batch:
        if item["init_status"] == "error":
            data = dict(id=item["id"], input=item["input"], output=item["truth"])
            all_init_errors.append(json.dumps(data)) 

eval_examples = []
with open(args.heldout_eval_file) as f:
    eval_lines = f.read().splitlines()
    for line in eval_lines:
        item = json.loads(line)
        data = dict(id=item["id"], input=item["input"], output=item["truth"])
        eval_examples.append(json.dumps(data)) 

# heldout_eval_file
    
with open(args.upstream_file) as f:
    upstream_lines = f.read().splitlines()

if args.ratio == 1:
    upstream_lines = upstream_lines
else:
    upstream_lines = random.sample(upstream_lines, len(all_init_errors)) # same number of examples

mixed_lines = upstream_lines + all_init_errors
with open(args.mixed_offline_file, "w") as f:
    for line in mixed_lines:
        f.write(line+"\n")
with open(args.mixed_offline_file.replace(".jsonl", ".dev.jsonl"), "w") as f:
    for line in eval_examples:
        f.write(line+"\n")

print(f"len(upstream_lines)={len(upstream_lines)}")
print(f"len(all_init_errors)={len(all_init_errors)}")
print(f"len(mixed_lines)={len(mixed_lines)}")
print(f"len(eval_examples)={len(eval_examples)}")