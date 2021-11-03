from argparse import Namespace
import argparse
from torch import detach
from semanticdebugger.models.utils import set_seeds
from semanticdebugger.debug_algs.cl_none import NoneCL
from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
from semanticdebugger.debug_algs.cl_online_ewc_alg import OnlineEWC
from semanticdebugger.debug_algs.offline_debug_bounds import OfflineDebugger
from semanticdebugger.debug_algs.cl_mbcl_alg import MemoryBasedCL
from semanticdebugger.debug_algs.index_based.cl_indexed_alg import IndexBasedCL
from semanticdebugger.debug_algs.cl_hypernet_alg import HyperCL
from semanticdebugger.debug_algs.distant_supervision import data_collection
import logging
import os
import json
from tqdm import tqdm
import numpy as np
import wandb

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

def setup_args(args):
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

    if args.cl_method_name == "none_cl":
        debugging_alg = NoneCL(logger=logger)
    elif args.cl_method_name == "simple_cl":
        debugging_alg = ContinualFinetuning(logger=logger)
    elif args.cl_method_name == "online_ewc":
        debugging_alg = OnlineEWC(logger=logger)
    elif args.cl_method_name == "offline_debug":
        debugging_alg = OfflineDebugger(logger=logger)
    elif args.cl_method_name in ["er", "mir"]:  # replay only
        assert args.replay_frequency > 0
        assert args.replay_size > 0
        if args.cl_method_name == "mir":
            args.use_mir = True
            assert args.replay_candidate_size >= args.replay_size
            assert args.num_adapt_epochs >= 1 # this is for the virtual update 
        else:
            assert args.num_adapt_epochs <= 0
        debugging_alg = MemoryBasedCL(logger=logger)
        debugging_alg.name = args.cl_method_name
    elif args.cl_method_name == "mbpa":
        assert args.num_adapt_epochs > 0
        assert args.replay_frequency <= 0
        assert args.replay_size <= 0
        debugging_alg = MemoryBasedCL(logger=logger)
        debugging_alg.name = args.cl_method_name
    elif args.cl_method_name == "mbpa++":
        assert args.num_adapt_epochs > 0
        assert args.replay_frequency > 0
        assert args.replay_size > 0
        debugging_alg = MemoryBasedCL(logger=logger)
        debugging_alg.name = args.cl_method_name
    elif args.cl_method_name == "index_cl":
        assert args.replay_frequency > 0
        assert args.replay_size > 0
        assert args.num_adapt_epochs <= 0
        debugging_alg = IndexBasedCL(logger=logger)
        debugging_alg.name = args.cl_method_name 
    elif args.cl_method_name == "hyper_cl":
        debugging_alg = HyperCL(logger=logger)
    elif args.cl_method_name == "simple_ds_mine":
        debugging_alg = data_collection.MiningSupervision(logger=logger)
    
    
    data_args = Namespace(
        submission_stream_data=args.submission_stream_data,
        upstream_eval_data=args.upstream_eval_data,
        heldout_submission_data=args.heldout_submission_data,
        upstream_data_path=args.upstream_data_path,
        # sampled_upstream_json_path=args.sampled_upstream_json_path,
        # pass_sample_size=args.pass_sample_size,
        do_lowercase=args.do_lowercase,
        append_another_bos=args.append_another_bos,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        task_name=args.task_name,
        result_file=args.result_file,
        train_batch_size=args.train_batch_size,
        predict_batch_size=args.predict_batch_size,
        num_beams=args.num_beams,
        max_timecode=args.max_timecode,
        accumulate_eval_freq=-1,
        # use_sampled_upstream=args.use_sampled_upstream,
    )

    base_model_args = Namespace(
        model_type=args.base_model_type,
        base_model_path=args.base_model_path
    )
    if args.cl_method_name in ["none_cl", "simple_cl", "online_ewc", "offline_debug", "er", "mir", "mbpa", "mbpa++", "index_cl", "hyper_cl", "simple_ds_mine"]:
        debugger_args = Namespace(
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            adam_epsilon=args.adam_epsilon,
            warmup_steps=0,
            total_steps=10000,
            num_epochs=args.num_train_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            save_ckpt_freq=args.save_ckpt_freq,
            ckpt_dir=args.ckpt_dir,
            skip_instant_eval=args.skip_instant_eval,
            kr_eval_freq=args.kr_eval_freq,
            kr_eval_mode=args.kr_eval_mode,
            okr_sample_size=args.okr_sample_size,
            okr_sample_seed=args.okr_sample_seed,
            kg_eval_freq=args.kg_eval_freq,
            kg_eval_mode=args.kg_eval_mode,
        )
        if args.cl_method_name == "online_ewc":
            setattr(debugger_args, "ewc_lambda", args.ewc_lambda)
            setattr(debugger_args, "ewc_gamma", args.ewc_gamma)       
        elif args.cl_method_name in ["er", "mbpa", "mbpa++", "mir", "index_cl"]: 
            setattr(debugger_args, "use_replay_mix", args.use_replay_mix)
            setattr(debugger_args, "replay_size", args.replay_size)
            setattr(debugger_args, "replay_candidate_size", args.replay_candidate_size)
            setattr(debugger_args, "replay_frequency", args.replay_frequency)
            setattr(debugger_args, "memory_path", args.memory_path)
            setattr(debugger_args, "init_memory_cache_path", args.init_memory_cache_path)
            setattr(debugger_args, "memory_key_encoder", args.memory_key_encoder)
            setattr(debugger_args, "memory_store_rate", args.memory_store_rate)
            setattr(debugger_args, "upstream_sample_ratio", args.upstream_sample_ratio) 
            setattr(debugger_args, "num_adapt_epochs", args.num_adapt_epochs)
            setattr(debugger_args, "inference_query_size", args.inference_query_size)
            setattr(debugger_args, "local_adapt_lr", args.local_adapt_lr)
            if args.cl_method_name == "mir" or args.use_mir:
                setattr(debugger_args, "mir_abalation_args", args.mir_abalation_args)  
            if args.cl_method_name == "index_cl":
                setattr(debugger_args, "use_mir", args.use_mir)  
                setattr(debugger_args, "index_rank_method", args.index_rank_method)  
                setattr(debugger_args, "indexing_method", args.indexing_method)  
                setattr(debugger_args, "indexing_args_path", args.indexing_args_path)                
                
        elif args.cl_method_name in ["hyper_cl"]:
            setattr(debugger_args, "adapter_dim", args.adapter_dim)
            setattr(debugger_args, "example_encoder_name", args.example_encoder_name)
            setattr(debugger_args, "task_emb_dim", args.task_emb_dim)
    return debugging_alg, data_args, base_model_args, debugger_args, logger


def run(args):
    debugging_alg, data_args, base_model_args, debugger_args, logger = setup_args(args)

    if args.num_threads_eval <= 0:
        # The Online Debugging Mode + Computing offline debugging bounds.
        
        # setattr(data_args, "data_stream_json_path", args.data_stream_json_path)
        # setattr(data_args, "replay_stream_json_path", args.replay_stream_json_path)
        debugging_alg.load_data(data_args)
    
        debugging_alg.load_base_model(base_model_args)
        debugging_alg.debugger_setup(debugger_args)

        if args.cl_method_name in ["offline_debug"]:
            debugging_alg.offline_debug()
            offline_bound_results = debugging_alg.single_timecode_eval(timecode=-1)
        else: 
            debugging_alg.online_debug() 

        
        # logger.info(f'output_info["final_eval_results"]={output_info["final_eval_results"]}')
        debugging_alg.save_result_file()
        logger.info(f"Finished. Results saved to {args.result_file}")
    else:
        # Parallel offline evaluation mode 
        timecodes = np.array_split(range(0, args.max_timecode+1), args.num_threads_eval)[args.current_thread_id]
        thread_results = {}
        debugging_alg.load_data(data_args)
        # debugging_alg.debugger_setup(debugger_args)
        for timecode in tqdm(timecodes, desc=f"Threads on {args.current_thread_id}"):
            logger.info(f"Starting the offline evaluation of {timecode}")
            timecode = int(timecode)
            base_model_args.base_model_path = os.path.join(args.ckpt_dir, f"model_ckpt_{timecode:03d}.pt")
            debugging_alg.debugger_args = debugger_args
            debugging_alg.load_base_model(base_model_args, mode="offline_eval")

            if args.cl_method_name in ["mbpa++"]:
                if args.num_adapt_epochs > 0:
                    debugging_alg.debugger_setup(debugger_args) # because there are local adaptation.
                else:
                    debugging_alg.debugger_args = debugger_args
            single_result = debugging_alg.single_timecode_eval(timecode)
            thread_results[timecode] = single_result
            # logger.info(f"Results: {json.dumps(single_result)}")
        with open(args.path_to_thread_result, "w") as f:
            json.dump(thread_results, f)
    return


def get_cli_parser():
    parser = argparse.ArgumentParser()

    # base_model_args
    parser.add_argument("--base_model_type",
                        default="facebook/bart-base", required=False)
    parser.add_argument(
        "--base_model_path",
        default="out/mrqa_squad_bart-base_1029_upstream_model//best-model.pt", type=str)

    # data_args

    parser.add_argument("--submission_stream_data",
                        default="/path/to/submission_stream")    

    # this will be used for evaluating forgetting
    parser.add_argument("--upstream_eval_data", 
                        default="experiments/eval_data/qa/upstream_eval.v1.jsonl")
                        
    parser.add_argument("--heldout_submission_data", 
                        default="experiments/eval_data/qa/heldout_eval.v1.json")

    parser.add_argument("--upstream_data_path",
                        default="data/mrqa_squad/mrqa_squad_train.jsonl")
                        # default="bug_data/mrqa_naturalquestions.sampled_upstream.jsonl")

    parser.add_argument("--task_name", default="mrqa")

    # base model args.
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--predict_batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument("--freeze_embeds", action='store_true', default=False)
    parser.add_argument('--max_input_length', type=int, default=888)
    parser.add_argument('--max_output_length', type=int, default=50)
    parser.add_argument("--append_another_bos", type=int,
                        default=1)  # should be true (1)


    # evalaution related
    parser.add_argument('--skip_instant_eval', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))

    parser.add_argument('--use_wandb', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))

    parser.add_argument('--kr_eval_freq', type=int, default=5)
    parser.add_argument('--kr_eval_mode', default="loss") # loss or metric

    parser.add_argument('--okr_sample_size', type=int, default=512)
    parser.add_argument('--okr_sample_seed', type=int, default=1337)
    
    parser.add_argument('--kg_eval_freq', type=int, default=5)
    parser.add_argument('--kg_eval_mode', default="loss") # loss or metric
    
    

    # feiw-benchmark
    # debugger_args

    parser.add_argument('--cl_method_name', type=str, default="none_cl",
                        help="the method name of the continual learning method")
    
    ### The HPs for Simple Continual Fine-tuning Method. ###
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
                        

    ### The HPs for Online EWC Method. ###
    parser.add_argument("--ewc_lambda", default=0.5, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--ewc_gamma", default=1, type=float,
                        help="Max gradient norm.")                        
    
    # parser.add_argument("--use_sampled_upstream", action='store_true', default=False)
 
    ### The HPs for replay-based methods and memory-based.
    parser.add_argument('--replay_size', type=int, default=8)
    parser.add_argument('--replay_candidate_size', type=int, default=8)
    parser.add_argument('--replay_frequency', type=int, default=1) # 1 means always replay for every steps, set to 10 means sample after 10 model updates.
    parser.add_argument('--memory_key_encoder', type=str, default="facebook/bart-base")
    parser.add_argument('--memory_path', type=str, default="")    
    parser.add_argument('--init_memory_cache_path', type=str, default="bug_data/memory_key_cache.pkl")
    parser.add_argument('--upstream_sample_ratio', type=float, default=-1)   #  
    parser.add_argument('--memory_store_rate', type=float, default=1.0)   # 1= always store all examples to the memory. 
    parser.add_argument('--num_adapt_epochs', type=int, default=1) #
    parser.add_argument('--inference_query_size', type=int, default=1) #
    parser.add_argument("--use_replay_mix", action='store_true', default=False) # mix the replayed examples with the current error examples.
    parser.add_argument('--local_adapt_lr', type=float, default=1e-5) #
    

    # MIR ablation options
    parser.add_argument('--mir_abalation_args', type=str, default="none")
    
    # Indexbased CL abalation options
    parser.add_argument('--use_mir', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--index_rank_method', type=str, default="most_similar")
    parser.add_argument('--indexing_method', type=str, default="bart_index")    # bart_index, biencoder 
    parser.add_argument('--indexing_args_path', type=str, default="exp_results/supervision_data/1012_dm_simple.train_args.json")    # bart_index, biencoder 
    

    ### The HPs for HyperCL
    parser.add_argument('--adapter_dim', type=int, default=32) # 1 means always replay for every steps, set to 10 means sample after 10 model updates.
    parser.add_argument('--example_encoder_name', type=str, default="roberta-base")
    parser.add_argument('--task_emb_dim', type=int, default=768)


    # To save all ckpts.


    # I/O parameters
    parser.add_argument('--prefix', type=str, default="",
                        help="Prefix for saving predictions")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument(
        "--result_file", default="bug_data/results.json", type=str)
    
    parser.add_argument("--ckpt_dir", type=str, default="experiments/ckpt_dirs/qa/nonecl",
                        help="path to all ckpts for saving")

    parser.add_argument("--save_ckpt_freq", type=int, default=5,    # 0 means no save for the intermidiate . but we always save the final model ckpt.
                        help="set to 1 if we want all ckpts and eval offline")
    # Offline Evaluation Mode in Parallel 

    parser.add_argument("--num_threads_eval", type=int, default=0,
                        help="0 means nothing; >0 means the number of gpu threads")
    parser.add_argument("--current_thread_id", type=int,
                        help="0 to num_threads_eval-1")
    parser.add_argument("--max_timecode", default=-1, type=int,
                        help="the maximum timecode to eval")
    parser.add_argument("--path_to_thread_result", type=str,
                        help="the path to save the thread results")

    return parser


if __name__ == '__main__':
    args = get_cli_parser().parse_args()

    if args.use_wandb:
        wandb_mode = "online"
    else:
        wandb_mode = "disabled"
    wandb_run = wandb.init(reinit=True, project="error-nlp", mode=wandb_mode, settings=wandb.Settings(start_method="fork"), name=args.prefix)
    run_name = wandb.run.name
    wandb.config.update(args) 
    run(args)
