
from argparse import Namespace
import argparse
from semanticdebugger.models.utils import set_seeds
from semanticdebugger.debug_algs.continual_finetune_alg import ContinualFinetuning
import logging
import os
import json
from tqdm import tqdm


class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # , file=sys.stderr)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def run(args):
    set_seeds(args.seed) 
    prefix = args.prefix
    log_filename = f"logs/{prefix}_online_debug.log"
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_filename),
                                  logging.StreamHandler(), TqdmHandler()])
    logger = logging.getLogger(__name__)
    
    logger.info(args)

    debugging_alg = ContinualFinetuning(logger=logger)

    data_args = Namespace(
        bug_stream_json_path=args.bug_stream_json_path,
        pass_pool_jsonl_path=args.pass_pool_jsonl_path,
        pass_sample_size=args.pass_sample_size,
        do_lowercase=args.do_lowercase,
        append_another_bos=args.append_another_bos,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        task_name=args.task_name,
        train_batch_size=args.train_batch_size,
        predict_batch_size=args.predict_batch_size,
        num_beams=args.num_beams,
    )

    base_model_args = Namespace(
        model_type=args.base_model_type,
        base_model_path=args.base_model_path
    )

    debugger_args = Namespace(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=0,
        total_steps=10000,
        num_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm
    )

    debugging_alg.load_data(data_args)
    debugging_alg.load_base_model(base_model_args)
    debugging_alg.debugger_setup(debugger_args)
    online_debug_results = debugging_alg.online_debug()

    output_info = {}
    output_info["method_class"] = debugging_alg.name
    output_info["base_model_args"] = str(debugging_alg.base_model_args)
    output_info["debugger_args"] = str(debugging_alg.debugger_args)
    output_info["data_args"] = str(debugging_alg.data_args)
    output_info["online_debug_results"] = online_debug_results
    output_info["sampled_passes_ids"] = [item["id"] for item in debugging_alg.sampled_passes]

    with open(args.result_file, "w") as f:
        json.dump(output_info, f)
    
    logger.info(f"Finished. Results saved to {args.result_file}")
 
    return
    

def get_cli_parser():
    parser = argparse.ArgumentParser()

    # base_model_args  
    parser.add_argument("--base_model_type", default="facebook/bart-base", required=False) 
    parser.add_argument("--base_model_path", default="out/mrqa_naturalquestions_bart-base_0617v4/best-model.pt", type=str) 

    # data_args

    parser.add_argument("--bug_stream_json_path", default="bug_data/mrqa_naturalquestions_dev.static_bug_stream.json")
    parser.add_argument("--pass_pool_jsonl_path", default="bug_data/mrqa_naturalquestions_dev.pass.jsonl")

    parser.add_argument("--task_name", default="mrqa_naturalquestions")
    parser.add_argument('--pass_sample_size', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--predict_batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=3)

    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument("--freeze_embeds", action='store_true', default=False)
    parser.add_argument('--max_input_length', type=int, default=888)
    parser.add_argument('--max_output_length', type=int, default=50)
    
    parser.add_argument("--append_another_bos", type=int, default=1) # should be true (1)


    # debugger_args 
    
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.") 
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.") 

    # I/O parameters 
    parser.add_argument('--prefix', type=str, default="nq_dev",
                        help="Prefix for saving predictions") 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--result_file", default="bug_data/results.json", type=str) 

    return parser

if __name__ == '__main__':
    args = get_cli_parser().parse_args()
    run(args)
