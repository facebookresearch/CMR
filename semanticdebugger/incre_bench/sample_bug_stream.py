import argparse
import logging
import json
import random
import numpy as np


def get_static_stream(bug_pool, batch_size, num_batches, use_score=False):
    assert batch_size * num_batches <= len(bug_pool)
    if use_score:
        # from easier to harder
        sorted_bugs = sorted(bug_pool, key=lambda x: x["score"]["QA-F1"], reverse=True)
    else:
        sorted_bugs = bug_pool
    static_bug_stream = []
    for start in range(0, len(bug_pool), batch_size):
        end = min(start + batch_size, len(bug_pool))
        bug_batch = sorted_bugs[start:end]
        static_bug_stream.append(sorted_bugs[start:end])
        if len(static_bug_stream) == num_batches:
            break
    return static_bug_stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bug_pool_file", default="bug_data/mrqa_naturalquestions_dev.bugs.jsonl", required=False)  # Input
    parser.add_argument(
        "--bug_strema_file", default="bug_data/mrqa_naturalquestions_dev.static_bug_stream.json", required=False)   # Output
    parser.add_argument("--batch_size", type=int, default=50, required=False)
    parser.add_argument("--num_batches", type=int, default=40, required=False)
    # batch_size * num_batches <= # lines of bug_pool_file
    args = parser.parse_args()

    bug_pool = []
    with open(args.bug_pool_file) as f:
        for line in f.read().splitlines():
            bug_pool.append(json.loads(line))
    if args.batch_size * args.num_batches > len(bug_pool):
        print("Error: args.batch_size ({}) * args.num_batches ({}) > # of bugs ({})".format(
            args.batch_size, args.num_batches, len(bug_pool)))
        return
    static_bug_stream = get_static_stream(
        bug_pool, args.batch_size, args.num_batches, use_score=True)
    with open(args.bug_strema_file, "w") as f:
        json.dump(static_bug_stream, f)
        
if __name__ == '__main__':
    main()
