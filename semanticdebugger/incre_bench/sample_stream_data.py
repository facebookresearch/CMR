import argparse
from enum import IntFlag
import logging
import json
import random
import numpy as np


def get_data_stream(data_pool, batch_size, num_batches, use_score=False):
    assert batch_size * num_batches <= len(data_pool)
    if use_score:
        # from easier to harder
        sorted_bugs = sorted(data_pool, key=lambda x: x["score"]["QA-F1"], reverse=True)
    else:
        # no sorting, randomly shuffuled
        random.shuffle(data_pool)
        sorted_bugs = data_pool
    data_stream = []
    for start in range(0, len(data_pool), batch_size):
        end = min(start + batch_size, len(data_pool))
        data_batch = sorted_bugs[start:end]
        data_stream.append(data_batch)
        if len(data_stream) == num_batches:
            break
    return data_stream




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bug_pool_file", default="bug_data/mrqa_naturalquestions_dev.bugs.jsonl", required=False)  # Input
    parser.add_argument(
        "--pass_pool_file", default="bug_data/mrqa_naturalquestions_dev.pass.jsonl", required=False)  # Input
    parser.add_argument(
        "--data_strema_file", default="bug_data/mrqa_naturalquestions_dev.data_stream.test.json", required=False)   # Output
    # parser.add_argument(
    #     "--sampled_pass_pool_file", default="bug_data/mrqa_naturalquestions_dev.sampled_pass.jsonl", required=False)   # Output
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--bug_sample_size", type=IntFlag, default=1000, required=False)
    parser.add_argument("--pass_sample_size", type=IntFlag, default=2200, required=False)
    parser.add_argument("--num_batches", type=int, default=100, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    # batch_size * num_batches <= # lines of bug_pool_file
    args = parser.parse_args()

    random.seed(args.seed)

    bug_pool = []
    with open(args.bug_pool_file) as f:
        for line in f.read().splitlines():
            bug_pool.append(json.loads(line))
    
    with open(args.pass_pool_file) as f:
        pass_pool = [json.loads(line) for line in f.read().splitlines()] 
    pass_pool = [item for item in pass_pool if item["score"]
                    ["EM"] == 1]  # only the EM=1 examples
    
    print(len(bug_pool), len(pass_pool))
    random.shuffle(pass_pool)
    random.shuffle(bug_pool)
    sampled_bug_pool = bug_pool[:args.bug_sample_size]
    sampled_pass_pool = pass_pool[:args.pass_sample_size]

    print(len(sampled_bug_pool), len(sampled_pass_pool))

    # if args.batch_size * args.num_batches > len(bug_pool):
    #     print("Error: args.batch_size ({}) * args.num_batches ({}) > # of bugs ({})".format(
    #         args.batch_size, args.num_batches, len(bug_pool)))
    #     return
    for item in sampled_bug_pool:
        item["init_status"] = "error"
    for item in sampled_pass_pool:
        item["init_status"] = "pass"

    sampled_data_pool = sampled_bug_pool + sampled_pass_pool

    data_stream = get_data_stream(
        sampled_data_pool, args.batch_size, args.num_batches, use_score=False)   # randomly sorted bugs
    with open(args.data_strema_file, "w") as f:
        json.dump(data_stream, f)

 
    # with open(args.sampled_pass_pool_file, "w") as f:
    #     for item in sample_examples:
    #         f.write(json.dumps(item) + "\n")
       
        
if __name__ == '__main__':
    main()
