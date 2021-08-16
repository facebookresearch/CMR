import json
import argparse
from semanticdebugger.task_manager.eval_metrics import evaluate_func

from sample_stream_data import generate_bugs
import random 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", default="data/mrqa_naturalquestions/mrqa_naturalquestions_train.jsonl", required=False)
    parser.add_argument(
        "--prediction_file", default="bug_data/mrqa_naturalquestions_train.predictions.jsonl", required=False)  # Input
    parser.add_argument(
        "--data_stream_file", default="bug_data/mrqa_naturalquestions_dev.data_stream.train.json", required=False)   # Output
    parser.add_argument(
        "--replay_stream_file", default="bug_data/mrqa_naturalquestions_dev.replay_stream.train.json", required=False)   # Output
    parser.add_argument(
        "--hidden_example_file", default="bug_data/mrqa_naturalquestions.hidden.jsonl", required=False)   # Output
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--replay_eval_size", type=int, default=-1, required=False)
    parser.add_argument("--bug_sample_size", type=int, default=1000, required=False)
    parser.add_argument("--pass_sample_size", type=int, default=2200, required=False)
    parser.add_argument("--hidden_sample_size", type=int, default=-1, required=False)
    parser.add_argument("--num_batches", type=int, default=100, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--metric", default="EM|QA-F1", required=False)

    parser.add_argument("--sample_method", default="no_replace", required=False)


      # batch_size * num_batches <= # lines of bug_pool_file
    args = parser.parse_args()

    random.seed(args.seed)
    
    print(args)
    
    
    # get the truth data
    
    all_truth_data = []
    for data_file in args.data_file.split("#"):
        truth_data = []
        with open(data_file) as fin:
            lines = fin.readlines()

        # train_examples = []
        for line in lines:
            # d = line.strip().split("\t")
            # truth_data.append((d[0], d[1:]))
            d = json.loads(line)
            truth_data.append((d["input"], d["output"], d["id"]))
        all_truth_data.append(truth_data)

    all_pred_data = []
    merged_restuls_all = None
    for prediction_file in args.prediction_file.split("#"):
        with open(prediction_file, "r") as f:
            predictions = json.load(f)
        # get evaluation results.
        print(f"len(predictions): {len(predictions)}")
        print(f"len(all_truth_data[len(all_pred_data)]): {len(all_truth_data[len(all_pred_data)])}")
        results, results_all = evaluate_func(
            predictions, all_truth_data[len(all_pred_data)], args.metric, return_all=True)
        print(f"{prediction_file}; Evaluation results: {results}")
        all_pred_data.append(predictions)
        if merged_restuls_all is None:
            merged_restuls_all = results_all
        else:
            for key in merged_restuls_all:
                merged_restuls_all[key].extend(results_all[key])


    merged_truth_data = []
    for item in all_truth_data:
        merged_truth_data.extend(item)
    merged_predictions = []
    for item in all_pred_data:
        merged_predictions.extend(item)

    

    bug_pool, pass_pool = generate_bugs(merged_predictions, merged_truth_data, merged_restuls_all)

    print(f"len(bug_pool)={len(bug_pool)}; len(pass_pool)={len(pass_pool)} <--- len(predictions)={len(merged_predictions)}")
    random.shuffle(pass_pool)
    random.shuffle(bug_pool)
    sampled_bug_pool = bug_pool[:args.bug_sample_size]
    sampled_pass_pool = pass_pool[:args.pass_sample_size]
    



if __name__ == '__main__':
    main()


"""

python semanticdebugger/benchmark_gen/sample_mixed_stream_data.py \
--sample_method with_replace \
--data_file \
data/mrqa_naturalquestions/mrqa_naturalquestions_dev.jsonl#\
data/mrqa_squad/mrqa_squad_dev.jsonl#\
data/mrqa_triviaqa/mrqa_triviaqa_dev.jsonl#\
data/mrqa_hotpotqa/mrqa_hotpotqa_dev.jsonl \
 --prediction_file \
bug_data/mrqa_naturalquestions_dev.predictions.jsonl#\
bug_data/mrqa_squad_dev.predictions.jsonl#\
bug_data/mrqa_triviaqa_dev.predictions.jsonl#\
bug_data/mrqa_hotpotqa_dev.predictions.jsonl \
 --data_stream_file exp_results/data_streams/mrqa.mixed.data_stream.test.json \
--batch_size 32 --num_batches 100 \
--seed 42 \
--bug_sample_size 1091 --pass_sample_size 2109

"""