
import json
import random
from semanticdebugger.benchmark_gen import sample_stream_data
from semanticdebugger.task_manager.eval_metrics import evaluate_func 




def create_training_stream(args):
    # setattr(data_args, "data_stream_json_path", args.data_stream_json_path)
    # setattr(data_args, "replay_stream_json_path", args.replay_stream_json_path)

    # with open(data_args.data_stream_json_path) as f:
    #     data_stream = json.load(f)

    upstream_truth_data = []
    with open(args.upstream_data_file) as fin:
        lines = fin.readlines()
    for line in lines:
        # d = line.strip().split("\t")
        # truth_data.append((d[0], d[1:]))
        d = json.loads(line)
        upstream_truth_data.append((d["input"], d["output"], d["id"])) 
    
    with open(args.upstream_data_prediction_file, "r") as f:
        M0_predictions = json.load(f)  

    print(f"len(predictions): {len(M0_predictions)}")
    print(f"len(upstream_truth_data]): {len(upstream_truth_data)}")
    results, results_all = evaluate_func(
        M0_predictions, upstream_truth_data , "EM|QA-F1", return_all=True)
    print(f"Upstream evaluation results: {results}")
    bug_pool, pass_pool = sample_stream_data.generate_bugs(M0_predictions, upstream_truth_data, results_all)

    sampled_M0_errors = random.sample(bug_pool, args.train_stream_length * args.train_stream_episode_size)
    sampled_init_memory = random.sample(pass_pool, args.init_memory_size)
    sampled_train_stream = sample_stream_data.get_data_stream(
            sampled_M0_errors, args.train_stream_episode_size, args.train_stream_length, use_score=False)
    # randomly sorted bugs
    return sampled_init_memory, sampled_train_stream
