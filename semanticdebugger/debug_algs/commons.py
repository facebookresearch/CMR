
import logging
import random
from semanticdebugger.task_manager.eval_metrics import evaluate_func
import torch
from transformers import BartTokenizer, BartConfig
import json
from tqdm import tqdm 


class OnlineDebuggingMethod():
    def __init__(self, logger=None):
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
        self.bug_stream = []
        self.bug_train_loaders = []
        self.bug_eval_loaders = []
        self.num_bug_batches = None
        self.bug_batch_size = None
        self.sampled_passes = []
        self.forget_eval_loader = None
        # utils
        self.use_cuda = torch.cuda.is_available()
        self.tokenizer = BartTokenizer.from_pretrained("bart-large")
        self.timecode = None
        self.metric = "EM|QA-F1"
        
        if self.use_cuda:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0
        return

    def _check_data_args(self):
        required_atts = ["bug_stream_json_path",
                         "pass_pool_jsonl_path",
                         "pass_sample_size",
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
        # Create data loaders
        for bug_batch in tqdm(self.bug_stream, desc="Creating the bug data loaders."):
            formatted_bug_batch = self.data_formatter(bug_batch)
            train_bug_dataloader, eval_bug_dataloader = self.get_dataloader(
                data_args, formatted_bug_batch, mode="both")
            self.bug_train_loaders.append(train_bug_dataloader)
            self.bug_eval_loaders.append(eval_bug_dataloader)
        assert len(self.bug_train_loaders) == self.num_bug_batches
        # Create loaders for the sampled pass examples
        with open(data_args.pass_pool_jsonl_path) as f:
            pass_pool = [json.loads(line) for line in set(f.read().splitlines())]
        # TODO: decide how to sample later.
        # random.shuffle(pass_pool)
        # sample the most correct ones.
        pass_pool.sort(key = lambda x: x["score"]["QA-F1"], reverse=True)
        sample_examples = pass_pool[:data_args.pass_sample_size]
        self.sampled_passes = sample_examples
        sample_examples = self.data_formatter(sample_examples)
        _, self.forget_eval_loader = self.get_dataloader(
            data_args, sample_examples, mode="eval")
        return

    def online_debug(self):
        self.logger.info("Start Online Debugging")
        self.logger.info(f"Number of Batches of Bugs: {self.num_bug_batches}")
        self.logger.info(f"Bug Batch Size: {self.bug_batch_size}")
        self.timecode = 0
        res_on_bugs = []
        res_on_passes = []
        for bug_train_loader, bug_eval_loader in tqdm(
                zip(self.bug_train_loaders, self.bug_eval_loaders), desc="Online Debugging", total=self.num_bug_batches):

            # Before Bug-Fixing
            bug_before_results = self.evaluate(bug_eval_loader)
            self.logger.info("-"*10+f"Timecode: {self.timecode}"+"-"*10)
            self.logger.info(
                f"Before Bug-fixing the results on bug-batch-{self.timecode} = {bug_before_results}")
            if len(res_on_passes) == 0:
                pass_before_results = self.evaluate(self.forget_eval_loader)
            else:
                pass_before_results = res_on_passes[-1][0]
            self.logger.info(
                f"Before Bug-fixing the results on the sampled pass cases = {pass_before_results}")

            # Fix the bugs by mini-batch based "training"
            self.logger.info("Start bug-fixing ....")
            self.fix_bugs(bug_train_loader)
            self.logger.info("Start bug-fixing .... Done!")

            # After Bug-Fixing
            bug_after_results = self.evaluate(bug_eval_loader)
            self.logger.info(
                f"After Bug-fixing the results on bug-batch-{self.timecode} = {bug_after_results}")
            pass_after_results = self.evaluate(self.forget_eval_loader)
            self.logger.info(
                f"After Bug-fixing the results on the sampled pass cases = {pass_after_results}")
            res_on_bugs.append((bug_before_results, bug_after_results))
            res_on_passes.append((pass_before_results, pass_after_results))
            self.timecode += 1
        return res_on_bugs, res_on_passes

    def evaluate(self, eval_dataloader=None):
        """Evaluates the performance"""
        if not eval_dataloader:
            eval_dataloader = self.bug_eval_loaders[self.timecode]
        predictions = self.base_model_infer(eval_dataloader)
        assert len(predictions) == len(eval_dataloader)
        predictions = [p.strip() for p in predictions]
        results = evaluate_func(predictions, eval_dataloader.data, self.metric)
        return results

    def base_model_infer(self, eval_dataloader):
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

    def load_base_model(self, base_model_args):
        raise NotImplementedError(
            "Please Implement the `load_base_model` method in your class.")

    def debugger_setup(self):
        raise NotImplementedError(
            "Please Implement the `debugger_setup` method in your class.")

    def fix_bugs(self, bug_batch):
        raise NotImplementedError("Please Implement the `fix_bugs` method in your class.")
