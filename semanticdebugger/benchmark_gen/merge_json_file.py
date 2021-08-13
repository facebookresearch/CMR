import json
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument(
        "--input_file_pattern",
        default="exp_results/data_streams/paraphrase/mrqa_naturalquestions_dev.data_stream.test.wr.para_data_*.json", type=str)
parser.add_argument(
        "--output_file",
        default="exp_results/data_streams/paraphrase/mrqa_naturalquestions_dev.data_stream.test.wr.para_data.json", type=str)
parser.add_argument(
        "--range",
        default="range(16)", type=str)

args = parser.parse_args()


all_data = None
for shard_id in eval(args.range):
    filename = args.input_file_pattern.replace("#", str(shard_id))
    with open(filename) as f:
        data = json.load(f)
    if all_data is None:
        all_data = data
    if type(all_data) == dict:
        all_data.update(data)
    else:
        all_data += data

with open(args.output_file, "w") as f:
    json.dump(all_data, f)

