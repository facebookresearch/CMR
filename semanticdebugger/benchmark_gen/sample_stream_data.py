import argparse 
import json
import random
 
from semanticdebugger.task_manager.eval_metrics import evaluate_func
import numpy as np



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
            bug_lines.append(item)
            item["init_status"] = "error"
        if em == True: 
            pass_lines.append(item)
            item["init_status"] = "pass"
    return bug_lines, pass_lines


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



def get_replay_stream(data_stream, replay_eval_size, window_size=10):
    past_error_pool = {} # errror in terms of the initial model 
    
    replay_stream = []
    for timecode, data_batch in enumerate(data_stream):
        # add the errors to the pool
        past_error_pool[timecode] = []
        for item in data_batch:
            if True or item["init_status"] == "error":
                past_error_pool[timecode].append(item)  
        
        # build the pool
        start_ind = max(0, timecode-window_size)
        end_ind = min(timecode, len(past_error_pool))
        candidate_replay_instances = []
        
        if end_ind == 0:
            continue # do not add for the first episode because there is no history for it

        for ind in range(start_ind, end_ind): # not including itself
            candidate_replay_instances += past_error_pool[ind]

        for _db in data_stream[-5:]:
            if len(candidate_replay_instances) >= replay_eval_size:
                break
            for item in _db:
                if len(candidate_replay_instances) >= replay_eval_size:
                    break
                if item["init_status"] == "pass":
                    candidate_replay_instances.append(item)
                                

        # print(start_ind, end_ind, len(candidate_replay_instances))
        assert len(candidate_replay_instances) >= replay_eval_size
        sampled_replay = random.sample(candidate_replay_instances, replay_eval_size)
        replay_stream.append(sampled_replay)
    
    assert len(replay_stream) == len(data_stream) - 1
    return replay_stream



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
    parser.add_argument("--replay_eval_size", type=int, default=32, required=False)
    parser.add_argument("--bug_sample_size", type=int, default=1000, required=False)
    parser.add_argument("--pass_sample_size", type=int, default=2200, required=False)
    parser.add_argument("--hidden_sample_size", type=int, default=-1, required=False)
    parser.add_argument("--num_batches", type=int, default=100, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--metric", default="EM|QA-F1", required=False)
    # batch_size * num_batches <= # lines of bug_pool_file
    args = parser.parse_args()

    random.seed(args.seed)
    
    
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

    with open(args.prediction_file, "r") as f:
        predictions = json.load(f)
    # get evaluation results.
    results, results_all = evaluate_func(
        predictions, truth_data, args.metric, return_all=True)
    print(f"Evaluation results: {results}")
    bug_pool, pass_pool = generate_bugs(predictions, truth_data, results_all)

    print(f"len(bug_pool)={len(bug_pool)}; len(pass_pool)={len(pass_pool)} <--- len(predictions)={len(predictions)}")

    # bug_pool = []
    # with open(args.bug_pool_file) as f:
    #     for line in f.read().splitlines():
    #         bug_pool.append(json.loads(line))
    
    # with open(args.pass_pool_file) as f:
    #     pass_pool = [json.loads(line) for line in f.read().splitlines()] 
    # pass_pool = [item for item in pass_pool if item["score"]
    #                 ["EM"] == 1]  # only the EM=1 examples
     
    random.shuffle(pass_pool)
    random.shuffle(bug_pool)
    sampled_bug_pool = bug_pool[:args.bug_sample_size]
    sampled_pass_pool = pass_pool[:args.pass_sample_size]
    if args.hidden_sample_size > 0 and args.hidden_sample_size + args.pass_sample_size <= len(pass_pool):
        hidden_examples = pass_pool[-args.hidden_sample_size:]
        with open(args.hidden_example_file, "w") as f:
            for item in hidden_examples:
                f.write(json.dumps(item) + "\n")

    print(len(sampled_bug_pool), len(sampled_pass_pool))
 
    sampled_data_pool = sampled_bug_pool + sampled_pass_pool

    data_stream = get_data_stream(
        sampled_data_pool, args.batch_size, args.num_batches, use_score=False)   # randomly sorted bugs

    
    replay_stream = get_replay_stream(data_stream, args.replay_eval_size)
    
    # replay_stream.insert(0, random.sample(sampled_bug_pool, args.replay_eval_size))
    replay_stream.insert(0, random.sample(sampled_data_pool, args.replay_eval_size))

    with open(args.data_stream_file, "w") as f:
        json.dump(data_stream, f)
    
    with open(args.replay_stream_file, "w") as f:
        json.dump(replay_stream, f)

 
   
       
        
if __name__ == '__main__':
    main()



"""
python semanticdebugger/benchmark_gen/sample_stream_data.py \
    --data_file data/mrqa_naturalquestions/mrqa_naturalquestions_train.jsonl \
    --prediction_file bug_data/mrqa_naturalquestions_train.predictions.jsonl \
    --data_stream_file exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.train.json \
    --hidden_example_file exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl \
    --replay_stream_file exp_results/data_streams/mrqa_naturalquestions_dev.replay_stream.train.json \
    --batch_size 32 --num_batches 500 \
    --bug_sample_size 4688 --pass_sample_size 11312 \
    --hidden_sample_size 500

# 500*32 - 4688 = 11312


python semanticdebugger/benchmark_gen/sample_stream_data.py \
    --data_file data/mrqa_naturalquestions/mrqa_naturalquestions_dev.jsonl \
    --prediction_file bug_data/mrqa_naturalquestions_dev.predictions.jsonl \
    --data_stream_file exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.json \
    --replay_stream_file exp_results/data_streams/mrqa_naturalquestions_dev.replay_stream.test.json \
    --batch_size 32 --num_batches 100 \
    --bug_sample_size 1091 --pass_sample_size 2109

"""