"""
This script serves:
- initializing the adapters of BART-Adapater models such that it has almost the same perf as the non-adapter one.
- pre-training the hypernetwork stuff.
"""



import enum
from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
from semanticdebugger.models import run_bart
from semanticdebugger.models.utils import convert_model_to_single_gpu
from semanticdebugger.models.bart_with_adapater import MyBartWithAdapter, BartWithAdapterConfig
from semanticdebugger.models.mybart import MyBart
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from semanticdebugger.models.utils import freeze_embeds, trim_batch, convert_model_to_single_gpu

import argparse
import json
import logging



def get_dataloader(args, logger):
    cf_alg = ContinualFinetuning(logger)
    
    def formatter(datastream):
        formatted_data = []
        for data_batch in datastream:
            for bug in data_batch:
                # if "id" not in bug:
                #     _id = len(formatted_bug_batch)
                _id = bug["id"]
                _input = bug["input"]
                _mistake = bug["mistake"] # the model prediction (if it is a bug, then it is the error prediction)
                _truth = bug["truth"]   # a list of answers
                formatted_data.append((_input, [_mistake], _id))    
        return formatted_data

    with open(args.train_path) as f:
        train_data_stream = json.load(f)
    
    with open(args.test_path) as f:
        eval_data_stream = json.load(f)

        # Note that we cannot use the truth for the examples, but the "mistakes" of the base model,
            # as our purpose is to train the bart_with_adatper to have same behaviors of the bart 
    formatted_train_data = formatter(train_data_stream)
    formatted_eval_data = formatter(eval_data_stream)[:500]
    
    # with open(args.test_path) as f:
    #     pass_examples = [json.loads(line)
    #                         for line in set(f.read().splitlines())] 
    # formatted_eval_data = [(item["input"], item["truth"], item["id"]) for item in pass_examples]  

    train_data_loader, _ = cf_alg.get_dataloader(args, formatted_train_data, mode="train")

    _, eval_data_loader = cf_alg.get_dataloader(args, formatted_eval_data, mode="eval")
    
    return train_data_loader, eval_data_loader

def init_adapters_for_bart(args, logger): 
    train_dataloader, test_dataloader = get_dataloader(args, logger) 

    # Init the models
    model_type = "facebook/bart-base"
    base_model_path = "out/mrqa_naturalquestions_bart-base_0617v4/best-model.pt"
    config = BartWithAdapterConfig.from_pretrained(model_type)
    config.adapter_dim = 64
    setattr(config, "init_std", args.adapter_init_std)
    bart_w_adapter = MyBartWithAdapter(config)
    mybart_model = MyBart.from_pretrained(model_type, state_dict=convert_model_to_single_gpu(torch.load(base_model_path)))
    bart_w_adapter.model.load_state_dict(mybart_model.model.state_dict(), strict=False)
    
    # bart_w_adapter = mybart_model   # TODO: test.

    # set up the GPU parallel.
    setattr(args, "use_cuda", True)
    setattr(args, "n_gpu", torch.cuda.device_count()) 
    if args.use_cuda:
        # Enable multi-gpu training.
        bart_w_adapter.to(torch.device("cuda"))
        logger.info("Moving to the GPUs.")
        if args.n_gpu > 1:
            bart_w_adapter = torch.nn.DataParallel(bart_w_adapter)
    


    args.total_steps = args.num_train_epochs * len(train_dataloader.dataloader)
    # Train the bart with adapter.
    no_decay = ['bias', 'LayerNorm.weight']

    # TODO: only for the adapters
    adapter_named_parameters = {}
    def get_adapter_param(layers, prefix="encode"):
        for ind, layer in enumerate(layers):
            layer_prefix = f"{prefix}_layer_{ind}."
            for n, p in layer.named_parameters():
                if "adapter" in n:
                    adapter_named_parameters[layer_prefix+n] = p
    get_adapter_param(bart_w_adapter.module.encoders(), prefix="encoder")
    get_adapter_param(bart_w_adapter.module.decoders(), prefix="decoder")
    print(adapter_named_parameters.keys()) 

    optimizer_grouped_parameters = [
        {'params': [p for n, p in adapter_named_parameters.items() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in adapter_named_parameters.items() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=args.total_steps)
    init_perf = run_bart.inference(bart_w_adapter if args.n_gpu == 1 else bart_w_adapter.module, test_dataloader, args=args, save_predictions=False, logger=logger)
    logger.info(f"init_perf: {init_perf}")
    best_dev_performance, best_model_state_dict = run_bart.train(
        args, logger, bart_w_adapter, train_dataloader, test_dataloader, optimizer, scheduler)
    logger.info(best_dev_performance)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", default=1e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adapter_init_std", default=1e-7, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--total_steps", default=-1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_period", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--num_beams', type=int, default=4)

                        
    parser.add_argument("--train_path", default="exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.train.json", type=str)
    # parser.add_argument("--test_path", default="exp_results/data_streams/mrqa_naturalquestions_dev.hidden_passes.jsonl", type=str)
    parser.add_argument("--test_path", default="exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.json", type=str)
    parser.add_argument("--output_dir", default="exp_results/dynamic_stream/sem_debugger/models/", type=str)

    args = parser.parse_args()

    # set up the data for pre-training the adapter.    
    setattr(args, "train_batch_size", 8)
    setattr(args, "predict_batch_size", 8)
    setattr(args, "task_name", "mrqa_naturalquestions")
    setattr(args, "do_lowercase", False)
    setattr(args, "freeze_embeds", False)
    setattr(args, 'max_input_length', 888)
    setattr(args, 'max_output_length', 50)
    setattr(args, 'append_another_bos', 1) 
    setattr(args, 'weight_decay', 0.01)
    setattr(args, 'warmup_steps', 50)
    setattr(args, 'wait_step', 10)
    setattr(args, "quiet", False)
    


    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,)
                        # handlers=[logging.FileHandler(log_filename),
                        #           logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    # logger.info(args)
    init_adapters_for_bart(args, logger)
