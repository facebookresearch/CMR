# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
from os import path
import random 
import json



from cmr.models.utils import set_seeds
from cmr.task_manager.eval_metrics import evaluate_func
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def formatting_initial_status(data_name, predictions, truth_data, results_all, task="qa"):
    assert len(predictions) == len(truth_data) == len(
        results_all["EM"]) == len(results_all["QA-F1"]) 
    formatted_data = []
    for p, t, em, f1 in zip(predictions, truth_data, results_all["EM"], results_all["QA-F1"]):
        item = dict()
        item["input"] = t[0]
        item["truth"] = t[1]
        item["id"] =  t[2]
        item["mistake"] = p.strip()
        if task == "qa":
            if 0.5 < f1 < 1 and em == False:
                # remove the false negative ones..
                continue
            item["score"] = {"EM": int(em == True), "QA-F1": float(f1)}
            item["data_name"] = data_name
            if em == False:  
                item["init_status"] = "error"
            else:  
                item["init_status"] = "pass"
        formatted_data.append(item)
    return formatted_data

def load_datasets(args): 
    if args.task_name == "QA":
        truth_paths = {
            # "squad-train": "data/mrqa_squad/mrqa_squad_train.jsonl",
            "squad": "data/mrqa_squad/mrqa_squad_dev.jsonl",   
            "nq": "data/mrqa_naturalquestions/mrqa_naturalquestions_dev.jsonl", #
            "trivia": "data/mrqa_triviaqa/mrqa_triviaqa_dev.jsonl",  
            "hotpot": "data/mrqa_hotpotqa/mrqa_hotpotqa_dev.jsonl",   
            "news": "data/mrqa_newsqa/mrqa_newsqa_dev.jsonl",   
            "search": "data/mrqa_searchqa/mrqa_searchqa_dev.jsonl",   
        }
        prediction_paths = {
            # "squad-train": "upstream_resources/qa_upstream_preds/mrqa_squad_train.predictions.json",   
            "squad": "upstream_resources/qa_upstream_preds/mrqa_squad_dev.predictions.json",   
            "nq": "upstream_resources/qa_upstream_preds/mrqa_naturalquestions_dev.predictions.json", 
            "trivia": "upstream_resources/qa_upstream_preds/mrqa_triviaqa_dev.predictions.json",  
            "hotpot": "upstream_resources/qa_upstream_preds/mrqa_hotpotqa_dev.predictions.json",   
            "news": "upstream_resources/qa_upstream_preds/mrqa_newsqa_dev.predictions.json", 
            "search": "upstream_resources/qa_upstream_preds/mrqa_searchqa_dev.predictions.json", 
        }
        upstream_data_name = "squad"
    elif args.task_name == "NLI":
        truth_paths = {
            # "squad-train": "data/mrqa_squad/mrqa_squad_train.jsonl",
            "snli": "data/snli/snli_validation.jsonl",   
            "multi_nli_matched": "data/multi_nli/multi_nli_validation_matched.jsonl", #
            "multi_nli_mismatched": "data/multi_nli/multi_nli_validation_mismatched.jsonl", #
            "scitail": "data/scitail/scitail_dev.jsonl",   
            "anli": "data/anli/anli_dev.jsonl",   
        }
        prediction_paths = {
            "snli": "upstream_resources/nli_upstream_preds/snli-snli_validation.predictions.json",   
            "multi_nli_matched": "upstream_resources/nli_upstream_preds/multi_nli-multi_nli_validation_matched.predictions.json", #
            "multi_nli_mismatched": "upstream_resources/nli_upstream_preds/multi_nli-multi_nli_validation_mismatched.predictions.json", #
            "scitail": "upstream_resources/nli_upstream_preds/scitail-scitail_dev.predictions.json",   
            "anli": "upstream_resources/nli_upstream_preds/anli-anli_dev.predictions.json",   
        }
        upstream_data_name = "snli"
    

    all_truth_data = {}
    submission_data = {}
    heldout_submission_data = {}
    upstream_sampled_data = []
    for data_name, data_file in truth_paths.items():  
        truth_data = []
        with open(data_file) as fin:
            lines = fin.readlines() 
        # train_examples = []
        for line in lines: 
            d = json.loads(line)
            truth_data.append((d["input"], d["output"], d["id"]))
        all_truth_data[data_name] = truth_data

    

    for data_name, prediction_file in prediction_paths.items():
        with open(prediction_file, "r") as f:
            predictions = json.load(f)
        # get evaluation results. 
        results, results_all = evaluate_func(
            predictions, all_truth_data[data_name], args.metric, return_all=True)
        print(f"{data_name} --- Evaluation results: {results}") 
        formatted_data = formatting_initial_status(data_name, predictions, all_truth_data[data_name], results_all)
        random.shuffle(formatted_data)

        if data_name == upstream_data_name:
            # random.sample(formatted_data, k=args.upstream_eval_size)
            upstream_sampled_data = formatted_data[:args.upstream_eval_size]
            submission_data[upstream_data_name] = formatted_data[args.upstream_eval_size:]
            # print(f"len(upstream_sampled_data])={len(upstream_sampled_data)}")
        else:
            heldout_submission_data[data_name] = formatted_data[:args.heldout_submission_size]    # held-out
            submission_data[data_name] = formatted_data[args.heldout_submission_size:]
            # print(f"len(heldout_submission_data['{data_name}'])={len(heldout_submission_data[data_name])}")
        print(f"len(submission_data['{data_name}'])={len(submission_data[data_name])}")
        

    for data_name, data in submission_data.items():
        num_examples = len(data) 
        error_nums = [1 for item in data if item["init_status"] == "error"]
        print(f"{data_name} -- # examples = {num_examples};  Error rate: {sum(error_nums)/num_examples}")
    
    
    # QA_submission_data, QA_heldout_submission_data, QA_upstream_sampled_data 
    return submission_data, heldout_submission_data, upstream_sampled_data


def generate_submission_stream(submission_data, args, cfg):
    submission_stream = []
    upstream = cfg["upstream"]; T = cfg["T"]; b = cfg["b"]
    alpha = cfg["alpha"]; beta = cfg["beta"]; gamma = cfg["gamma"]
    assert upstream in submission_data
    OODs = [data_name for data_name in submission_data if data_name != upstream]
    N = len(OODs) # except for the upstream data
    
    if beta == 1:
        if args.task_name.lower() == "qa":
            current_major_ood = "nq"
    else:
        current_major_ood = random.choice(OODs)  # the initial major OOD cluster 
    for t in range(1, T+1):
        S_t = []
        if alpha == 0:
            b_upstream = 0 # special case when upstream data ratio = 0; (because 0^0=1 by definition)
        else:
            b_upstream = round(b * (alpha**(t-1))) 
        b_ood = b - b_upstream
        b_ood_major = round(b_ood * gamma)
        b_ood_diverse = b_ood - b_ood_major
        S_t += random.sample(submission_data[upstream], k=b_upstream)
        S_t += random.sample(submission_data[current_major_ood], k=b_ood_major)
        other_oods = [o for o in OODs if o != current_major_ood]
        # diverse_pools = []
        for o in other_oods: 
            # diverse_pools += submission_data[o]
            S_t += random.sample(submission_data[o], k=int(b_ood_diverse/len(other_oods)))

        if len(S_t) < b:
            o = random.choice(other_oods)
            S_t += random.sample(submission_data[o], k=b-len(S_t))
        assert len(S_t) == b
        # deal with the buffer 
        # Switch major ood 
        if random.random() < 1 - beta:
            current_major_ood = random.choice(other_oods)
        submission_stream.append(S_t)
    
    # visualize_stream(submission_stream, [upstream] + OODs, cfg, args) 
    return submission_stream    


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--upstream_eval_size", type=int, default=512)
    parser.add_argument("--heldout_submission_size", type=int, default=256)
    parser.add_argument("--episode_size", type=int, default=64)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="EM|QA-F1")
    parser.add_argument("--num_val", type=int, default=3)
    parser.add_argument("--num_test", type=int, default=5)
    parser.add_argument("--submission_stream_file", default="experiments/eval_data/qa/submission_stream.#args.json")
    parser.add_argument("--sampled_upstream_dataset", default="experiments/eval_data/qa/upstream_eval.jsonl")
    parser.add_argument("--heldout_submission_eval_file", default="experiments/eval_data/qa/heldout_eval.jsonl")
    parser.add_argument("--task_name", default="QA")
    
    
    args = parser.parse_args()
    print(args)
    set_seeds(args.seed)

    if args.task_name == "NLI":
        args.submission_stream_file = args.submission_stream_file.replace("qa", "nli")
        args.sampled_upstream_dataset = args.sampled_upstream_dataset.replace("qa", "nli")
        args.heldout_submission_eval_file = args.heldout_submission_eval_file.replace("qa", "nli") 

     # QA:
    configs = {}
    # configs["QA"] = dict(upstream="squad", T=args.num_episodes, b=args.episode_size, alpha=0.98, beta=1, gamma=1)
    configs["QA"] = []
    configs["QA"].append(dict(upstream="squad", T=args.num_episodes, b=args.episode_size, alpha=0.9, beta=0.9, gamma=0.8))
    configs["QA"].append(dict(upstream="squad", T=args.num_episodes, b=args.episode_size, alpha=0.9, beta=0.5, gamma=0.8))
    configs["QA"].append(dict(upstream="squad", T=args.num_episodes, b=args.episode_size, alpha=0.9, beta=0.1, gamma=0.8)) 
  
    configs["QA"].append(dict(upstream="squad", T=args.num_episodes, b=args.episode_size, alpha=0.9, beta=0.5, gamma=0.5)) 
    configs["QA"].append(dict(upstream="squad", T=args.num_episodes, b=args.episode_size, alpha=0.9, beta=0.5, gamma=0.2)) 
 
    configs["QA"].append(dict(upstream="squad", T=args.num_episodes, b=args.episode_size, alpha=0.1, beta=0.5, gamma=0.8))
    configs["QA"].append(dict(upstream="squad", T=args.num_episodes, b=args.episode_size, alpha=0.95, beta=0.5, gamma=0.8))

    

    configs["NLI"] = []
    configs["NLI"].append(dict(upstream="snli", T=args.num_episodes, b=args.episode_size, alpha=0.9, beta=0.9, gamma=0.8))
    configs["NLI"].append(dict(upstream="snli", T=args.num_episodes, b=args.episode_size, alpha=0.9, beta=0.5, gamma=0.8))
    configs["NLI"].append(dict(upstream="snli", T=args.num_episodes, b=args.episode_size, alpha=0.9, beta=0.1, gamma=0.8))
    

    

    # if args.task_name == "QA":
    submission_data, heldout_submission_data, upstream_sampled_data = load_datasets(args)
    

    with open(args.heldout_submission_eval_file, "w") as f:
        flat_heldout_submission_data =  []
        for v in list(heldout_submission_data.values()):
            flat_heldout_submission_data += v 
        for item in flat_heldout_submission_data:
            f.write(json.dumps(item) + "\n")

    with open(args.sampled_upstream_dataset, "w") as f:
        for item in upstream_sampled_data:
            f.write(json.dumps(item) + "\n")
    

    cfgs = configs[args.task_name]
    for cfg in cfgs:
        # Generate Validaiton/Test Streams
        validation_streams = []
        test_streams = []
        for _ in range(args.num_val):
            submission_stream = generate_submission_stream(submission_data, args, cfg)
            validation_streams.append(submission_stream)
        for _ in range(args.num_test):
            submission_stream = generate_submission_stream(submission_data, args, cfg)
            test_streams.append(submission_stream)
        prefix_title_str = f"T={cfg['T']},b={cfg['b']},alpha={cfg['alpha']},beta={cfg['beta']},gamma={cfg['gamma']}"

        title_str = prefix_title_str + "-val"
        with open(args.submission_stream_file.replace("#args", title_str), "w") as f:
            print(f"To save {f.name}")
            json.dump(validation_streams, f)

        title_str = prefix_title_str + "-test"
        with open(args.submission_stream_file.replace("#args", title_str), "w") as f:
            print(f"To save {f.name}")
            json.dump(test_streams, f) 

if __name__ == '__main__':
    main()


"""
python cmr/benchmark_gen/sample_submission_streams.py --task_name QA
python cmr/benchmark_gen/sample_submission_streams.py --task_name NLI --episode_size 256
"""