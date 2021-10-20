
import json
import random
from semanticdebugger.benchmark_gen import sample_stream_data
from semanticdebugger.task_manager.eval_metrics import evaluate_func 




def create_training_stream(args, logger):
    assert not args.use_dev_stream
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

    logger.info(f"len(predictions): {len(M0_predictions)}")
    logger.info(f"len(upstream_truth_data]): {len(upstream_truth_data)}")
    results, results_all = evaluate_func(
        M0_predictions, upstream_truth_data , "EM|QA-F1", return_all=True)
    logger.info(f"Upstream evaluation results: {results}")
    bug_pool, pass_pool = sample_stream_data.generate_bugs(M0_predictions, upstream_truth_data, results_all, f1_upper_bound=1.0)
    
    logger.info(f"len(bug_pool)={len(bug_pool)}")
    logger.info(f"len(pass_pool)={len(pass_pool)}")

    # TODO: add some pass_pool examples in bug pool?

    sampled_M0_errors = random.sample(bug_pool, args.train_stream_length * args.train_stream_episode_size)
    sampled_init_memory = random.sample(pass_pool, args.init_memory_size)
    sampled_train_stream = sample_stream_data.get_data_stream(
            sampled_M0_errors, args.train_stream_episode_size, args.train_stream_length, use_score=False)
    # randomly sorted bugs
    return sampled_init_memory, sampled_train_stream

def create_training_stream_with_dev(args, logger):
    assert args.use_dev_stream

    dev_memory = []
    with open(args.dev_memory) as f:
        for line in f.read().splitlines():
            d = json.loads(line)
            dev_memory.append(d)
        

    sampled_init_memory = random.sample(dev_memory, args.init_memory_size)

    with open(args.dev_stream) as f:
        dev_stream = json.load(f)
    dev_stream_examples = []
    # print(len(dev_stream))
    for batch in dev_stream:
        for item in batch:
            # print(item.keys())
            dev_stream_examples.append(item)

    # print(dev_stream_examples[:3])
    # print(len(dev_stream_examples))
    random.shuffle(dev_stream_examples)
    sampled_M0_errors = random.sample(dev_stream_examples, args.train_stream_length * args.train_stream_episode_size)    

    sampled_train_stream = sample_stream_data.get_data_stream(
            sampled_M0_errors, args.train_stream_episode_size, args.train_stream_length, use_score=False)
    # randomly sorted bugs
    return sampled_init_memory, sampled_train_stream