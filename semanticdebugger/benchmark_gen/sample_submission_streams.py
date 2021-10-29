import argparse
from os import path
import random 
import json
from typing_extensions import OrderedDict

from torch import numel
from semanticdebugger.benchmark_gen.bb_utils import bb_sample, bb_rescale, build_submission_stream
from semanticdebugger.notebooks.draw_utils import draw_curve, draw_stacked_bars
from semanticdebugger.task_manager.eval_metrics import evaluate_func
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def formatting_initial_status(data_name, predictions, truth_data, results_all, f1_upper_bound=0.5):
    assert len(predictions) == len(truth_data) == len(
        results_all["EM"]) == len(results_all["QA-F1"]) 
    formatted_data = []
    for p, t, em, f1 in zip(predictions, truth_data, results_all["EM"], results_all["QA-F1"]):
        item = dict()
        item["input"] = t[0]
        item["truth"] = t[1]
        item["id"] =  t[2]
        item["mistake"] = p.strip()
        item["score"] = {"EM": int(em == True), "QA-F1": float(f1)}
        item["data_name"] = data_name
        if em == False and f1 <= f1_upper_bound:  
            item["init_status"] = "error"
        else:  
            item["init_status"] = "pass"
        formatted_data.append(item)
    return formatted_data

def load_QA_datasets(args):
    all_data = {}
    truth_paths = {
        "nq": "data/mrqa_naturalquestions/mrqa_naturalquestions_dev.jsonl", # V_0
        "squad": "data/mrqa_squad/mrqa_squad_dev.jsonl",   # V_1
        "trivia": "data/mrqa_triviaqa/mrqa_triviaqa_dev.jsonl",  # V_2
        "hotpot": "data/mrqa_hotpotqa/mrqa_hotpotqa_dev.jsonl",     # More 
        "news": "data/mrqa_newsqa/mrqa_newsqa_dev.jsonl",     # More 
        "search": "data/mrqa_searchqa/mrqa_searchqa_dev.jsonl",     # More 
    }
    prediction_paths = {
        "nq": "upstream_resources/qa_upstream_preds/mrqa_naturalquestions_dev.predictions.json", # V_0
        "squad": "upstream_resources/qa_upstream_preds/mrqa_squad_dev.predictions.json",   # V_1
        "trivia": "upstream_resources/qa_upstream_preds/mrqa_triviaqa_dev.predictions.json",  # V_2
        "hotpot": "upstream_resources/qa_upstream_preds/mrqa_hotpotqa_dev.predictions.json",     # More 
        "news": "upstream_resources/qa_upstream_preds/mrqa_newsqa_dev.predictions.json", 
        "search": "upstream_resources/qa_upstream_preds/mrqa_searchqa_dev.predictions.json", 
    }

    all_truth_data = {}
    submission_data = {}
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
        submission_data[data_name] = formatted_data
    for data_name, data in submission_data.items():
        num_examples = len(data) 
        error_nums = [1 for item in data if item["init_status"] == "error"]
        print(f"{data_name} -- # examples = {num_examples};  Error rate: {sum(error_nums)/num_examples}")
    
    return submission_data





def generate_submission_stream(submission_data, args):
    # QA:
    configs = {}
    configs["QA"] = OrderedDict({
        "nq": dict(count=4000, a=1, b=1.3, upstream=True),
        "trivia": dict(count=1000, a=1.3, b=2, upstream=False),
        "hotpot": dict(count=1000, a=30, b=60, upstream=False),
        "squad": dict(count=1000, a=300, b=400, upstream=False), 
        "news": dict(count=1000, a=5, b=2, upstream=False),
        "search": dict(count=1000, a=4, b=1, upstream=False), 
    })
    

    all_pmfs = []
    for data_name, params in configs["QA"].items():
        if params["upstream"]:
            data_name = f"*{data_name}"
        all_pmfs += bb_sample(n=args.num_episodes, 
                                a=params["a"], b=params["b"], 
                                prefix=data_name, count=params["count"])
    
    all_pmfs_pd = pd.DataFrame(all_pmfs)
    # print(all_pmfs_pd.head())
    fig1 = draw_curve(df=all_pmfs_pd, fig_title="Temporal distribution of each data cluster.", y_scale=[0., 150], x_key="time_step", y_key="p", y_title="# of Examples")
    fig1.save('figures/mrqa.submission_stream_each.png')


    scaled_all_pmfs = bb_rescale(all_pmfs, batch_size=args.episode_size)
    scaled_all_pmfs_pd = pd.DataFrame(scaled_all_pmfs)
    fig2 =  draw_stacked_bars(df=scaled_all_pmfs_pd, fig_title="Submission Stream (bsz=64)", y_scale=[0., 65], x_key="time_step", y_key="sum(p)", y_title="# of Examples")
    fig2.save('figures/mrqa.submission_stream_all_scaled.png')

    submission_stream, init_error_pmfs = build_submission_stream(submission_data, scaled_all_pmfs, configs["QA"], args)

    init_error_pmfs_pd = pd.DataFrame(init_error_pmfs)
    fig3 =  draw_stacked_bars(df=init_error_pmfs_pd, fig_title="Error Stream of f_0 (bsz=64)", y_scale=[0., 65], x_key="time_step", y_key="sum(p)", y_title="# of Errors")
    fig3.save('figures/mrqa.submission_stream.init_errors.png')

    return submission_stream



def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--episode_size", type=int, default=64, required=False)
    parser.add_argument("--num_episodes", type=int, default=100, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--metric", default="EM|QA-F1", required=False)
    parser.add_argument("--submission_stream_file", default="exp_results/data_streams/mrqa.dynamic_stream.json", required=False)
    

    args = parser.parse_args()

    print(args)

    random.seed(args.seed)
    QA_submission_data = load_QA_datasets(args)
    submission_stream = generate_submission_stream(QA_submission_data, args)
    with open(args.submission_stream_file, "w") as f:
        json.dump(submission_stream, f)

if __name__ == '__main__':
    main()