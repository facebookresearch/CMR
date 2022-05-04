# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument(
        "--input_file_pattern",
        default="exp_results/data_streams/paraphrase/mrqa_naturalquestions_dev.data_stream.test.wr.para_data_#.json", type=str)
parser.add_argument(
        "--output_file",
        default="exp_results/data_streams/paraphrase/mrqa_naturalquestions_dev.data_stream.test.wr.para_data.json", type=str)
parser.add_argument(
        "--range",
        default="range(16)", type=str)
parser.add_argument(
        "--mode",
        default="json", type=str)

args = parser.parse_args()


all_data = None
for shard_id in eval(args.range):
    filename = args.input_file_pattern.replace("#", str(shard_id))
    if args.mode == "json":
        with open(filename) as f:
            print(f.name)
            data = json.load(f)
    elif args.mode == "jsonl":
        with open(filename) as f:
            print(f.name)
            data = [json.loads(line) for line in f.read().splitlines() if line]
    if all_data is None:
        all_data = data
    else:
        if type(all_data) == dict:
            all_data.update(data)
        else:
            all_data += data



with open(args.output_file, "w") as f:
    if args.mode == "json":
        json.dump(all_data, f)
    elif args.mode == "jsonl":
        for item in all_data:
            f.write(json.dumps(item) + "\n")