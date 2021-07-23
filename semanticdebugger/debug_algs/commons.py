
import logging
import random
from semanticdebugger.task_manager.eval_metrics import evaluate_func
import torch
from transformers import BartTokenizer, BartConfig
import json
from tqdm import tqdm
import os


class OnlineDebuggingMethod():
    def __init__(self, logger=None):
        self.name = "base_class"
        # logger
        self.logger = logger
        # args
        self.debugger_args = None
        self.base_model_args = None
        self.data_args = None
        # modules
        self.base_model = None
        self.debugger = None
        # data
        self.num_bug_batches = None
        self.bug_batch_size = None
        self.bug_stream = []
        self.bug_train_loaders = []
        self.bug_eval_loaders = []
        self.bug_all_train_loader = None
        self.bug_all_eval_loader = None        
        
        self.sampled_passes = []
        self.forget_eval_loader = None

        self.all_bug_examples = []
        self.sampled_upstream_examples = []

        # utils
        self.use_cuda = torch.cuda.is_available()
        self.tokenizer = BartTokenizer.from_pretrained("bart-large")
        self.timecode = None
        self.metric = "EM|QA-F1"

        if self.use_cuda:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0

        self.model_update_steps = 0 # number of updates over the base model.
        return

    def _check_data_args(self):
        required_atts = ["bug_stream_json_path",
                         "pass_pool_jsonl_path",
                         "sampled_upstream_json_path",
                         "do_lowercase",
                         "append_another_bos",
                         "max_input_length",
                         "max_output_length",
                         "task_name",
                         "num_beams"]
        assert all([hasattr(self.data_args, att) for att in required_atts])
        return

    def load_data(self, data_args):
        self.data_args = data_args
        self._check_data_args()
        # Load bug stream
        with open(data_args.bug_stream_json_path) as f:
            bug_stream = json.load(f)
        self.bug_stream = bug_stream
        self.num_bug_batches = len(bug_stream)
        self.bug_batch_size = len(bug_stream[0])
        # Create data loaders for each error batch.
        all_formatted_bugs = []
        for bug_batch in tqdm(self.bug_stream, desc="Creating the bug data loaders."):
            formatted_bug_batch = self.data_formatter(bug_batch)
            all_formatted_bugs += formatted_bug_batch
            train_bug_dataloader, eval_bug_dataloader = self.get_dataloader(
                data_args, formatted_bug_batch, mode="both")
            self.bug_train_loaders.append(train_bug_dataloader)
            self.bug_eval_loaders.append(eval_bug_dataloader)
        assert len(self.bug_train_loaders) == self.num_bug_batches
        self.all_bug_examples = all_formatted_bugs
        # Create the all bug loaders.
        self.bug_all_train_loader, self.bug_all_eval_loader = self.get_dataloader(
            data_args, all_formatted_bugs, mode="both")

        if data_args.pass_pool_jsonl_path:
            # Create loaders for the sampled pass examples
            with open(data_args.pass_pool_jsonl_path) as f:
                pass_examples = [json.loads(line)
                                 for line in set(f.read().splitlines())]
            self.sampled_passes = pass_examples
            pass_examples = self.data_formatter(pass_examples)
            _, self.forget_eval_loader = self.get_dataloader(
                data_args, pass_examples, mode="eval")

        if data_args.sampled_upstream_json_path:
            # Create loaders for the sampled pass examples
            with open(data_args.sampled_upstream_json_path) as f:
                sampled_upstream_examples = [json.loads(line)
                                          for line in set(f.read().splitlines())]
            self.sampled_upstream_examples = self.upstream_data_formatter(sampled_upstream_examples)
            # self.sampled_upstream_trainloader, self.sampled_upstream_evalloader = self.get_dataloader(
            #     data_args, sampled_upstream_examples, mode="eval")

        return

    def online_debug(self):
        self.logger.info("Start Online Debugging")
        self.logger.info(f"Number of Batches of Bugs: {self.num_bug_batches}")
        self.logger.info(f"Bug Batch Size: {self.bug_batch_size}")
        self.timecode = 0

        if self.debugger_args.save_all_ckpts:
            # save the initial model as the 0-th model.
            self._save_base_model()

        for bug_train_loader in tqdm(self.bug_train_loaders, desc="Online Debugging", total=self.num_bug_batches):
            ############### CORE ###############
            # Fix the bugs by mini-batch based "training"
            self.logger.info(f"Start bug-fixing .... Timecode: {self.timecode}")
            self.fix_bugs(bug_train_loader)   # for debugging
            self.logger.info("Start bug-fixing .... Done!")
            ############### CORE ###############
            self.timecode += 1
            if self.debugger_args.save_all_ckpts:
                self._save_base_model()
                # Note that we save the model from the id=1.
                # So the 0-th checkpoint should be the original base model.

    def single_timecode_eval(self, timecode):
        """Used only for offline eval of a single checkpoint of a specific timecode."""
        self.timecode = timecode
        result_dict = {}   # initialize for the given time code

        def _pack_as_dict(predictions, results, results_all):
            return {"predictions": predictions, "metric_results": results, "metric_results_detailed": results_all}

        self.logger.info("Start the Overall Error-Fixing Results....")
        # Overall Error-Fixing Results
        eval_results_overall_bug = self.evaluate(
            self.bug_all_eval_loader, verbose=True)
        result_dict["eval_results_overall_bug"] = _pack_as_dict(
            *eval_results_overall_bug)
        self.logger.info("Start the Overall Error-Fixing Results....Done")

        self.logger.info(
            "Start the Overall Forgetting Results (Knowledge Retain Acc)....")
        # Overall Forgetting Results (Knowledge Retain Acc)
        eval_results_overall_forget = self.evaluate(
            self.forget_eval_loader, verbose=True)
        result_dict["eval_results_overall_forget"] = _pack_as_dict(
            *eval_results_overall_forget)
        self.logger.info(
            "Start the Overall Forgetting Results (Knowledge Retain Acc)....Done")

        if self.name == "offline_debug":
            # only overall evaluation for the offline debugging.
            return result_dict

        # Error-Fixing performance on the current batch of errors.
        if self.timecode > 0:
            self.logger.info(
                "Start Error-Fixing performance on the Current batch of errors.....")
            bug_eval_loader = self.bug_eval_loaders[self.timecode-1]
            eval_results_current_errors = self.evaluate(bug_eval_loader)
            result_dict["eval_results_current_errors"] = _pack_as_dict(
                *eval_results_current_errors)
            self.logger.info(
                "Start Error-Fixing performance on the Current batch of errors.....Done")

        # Error-Fixing performance on the next batch of errors. (for the computation of real responsive efr)
        if self.timecode < len(self.bug_eval_loaders):
            self.logger.info(
                "Start Error-Fixing performance on the Next batch of errors.....")
            bug_eval_loader = self.bug_eval_loaders[self.timecode]
            eval_results_next_errors = self.evaluate(bug_eval_loader)
            result_dict["eval_results_next_errors"] = _pack_as_dict(
                *eval_results_next_errors)
            self.logger.info(
                "Start Error-Fixing performance on the Next batch of errors.....Done")

        return result_dict

    def _save_base_model(self, ckpt_name=None):
        output_dir = self.debugger_args.overtime_ckpt_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        model_state_dict = {k: v.cpu() for (
            k, v) in self.base_model.state_dict().items()}
        if ckpt_name:
            model_path = os.path.join(output_dir, f"model_ckpt_{ckpt_name}.pt")
        else:
            model_path = os.path.join(
                output_dir, f"model_ckpt_{self.timecode:03d}.pt")
        torch.save(model_state_dict, model_path)
        self.logger.info(f"Model saved to {model_path}.")

    def evaluate(self, eval_dataloader=None, verbose=False):
        """Evaluates the performance"""
        if not eval_dataloader:
            eval_dataloader = self.bug_eval_loaders[self.timecode]
        predictions = self.base_model_infer(eval_dataloader, verbose)
        assert len(predictions) == len(eval_dataloader)
        predictions = [p.strip() for p in predictions]
        results, return_all = evaluate_func(
            predictions, eval_dataloader.data, self.metric, return_all=True)

        return predictions, results, return_all

    def base_model_infer(self, eval_dataloader, verbose):
        raise NotImplementedError(
            "Please Implement the `base_model_infer` method in your class.")

    def check_debugger_args(self):
        raise NotImplementedError(
            "Please Implement the `check_debugger_args` method in your class.")

    def data_formatter(self, bug_batch):
        raise NotImplementedError(
            "Please Implement the `data_formatter` method in your class.")

    def get_dataloader(self, data_args, formatted_bug_batch):
        raise NotImplementedError(
            "Please Implement the `get_dataloader` method in your class.")

    def load_base_model(self, base_model_args, mode="online_debug"):
        raise NotImplementedError(
            "Please Implement the `load_base_model` method in your class.")

    def debugger_setup(self):
        raise NotImplementedError(
            "Please Implement the `debugger_setup` method in your class.")

    def fix_bugs(self, bug_loader, quiet=True):
        raise NotImplementedError(
            "Please Implement the `fix_bugs` method in your class.")

    def upstream_data_formatter(self, examples):
        # The continual fine-tuning method only uses the correct answers for fixing bugs.
        formatted_examples = []
        for example in examples: 
            _id = example["id"]
            _input = example["input"] 
            _truth = example["output"]   # a list of answers
            formatted_examples.append((_input, _truth, _id))
        return formatted_examples