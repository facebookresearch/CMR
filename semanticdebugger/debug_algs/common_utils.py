
import logging
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
        self.bug_data_args = None
        # modules
        self.base_model = None
        self.debugger = None
        self.bug_stream = []
        self.bug_train_loaders = []
        self.bug_eval_loaders = []
        self.num_bug_batches = None
        self.bug_batch_size = None
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

    def load_bug_streams(self, bug_data_args):
        required_atts = ["bug_stream_json_path", "do_lowercase", "append_another_bos",
                         "max_input_length", "max_output_length", "task_name", "num_beams"]
        assert all([hasattr(bug_data_args, att) for att in required_atts])
        with open(bug_data_args.bug_stream_json_path) as f:
            bug_stream = json.load(f)
        self.bug_stream = bug_stream
        self.num_bug_batches = len(bug_stream)
        self.bug_batch_size = len(bug_stream[0])
        for bug_batch in tqdm(self.bug_stream, desc="Creating the bug data loaders."):
            formatted_bug_batch = self.bug_formatter(bug_batch)
            train_bug_dataloader, eval_bug_dataloader = self.get_dataloader(
                bug_data_args, formatted_bug_batch, mode="both")
            self.bug_train_loaders.append(train_bug_dataloader)
            self.bug_eval_loaders.append(eval_bug_dataloader)
        return

    def online_debug(self):
        self.logger.info("Start Online Debugging")
        self.logger.info(f"Number of Batches of Bugs: {self.num_bug_batches}")
        self.logger.info(f"Bug Batch Size: {self.bug_batch_size}")
        self.timecode = 0
        for bug_batch in tqdm(self.bug_train_loaders, desc="Online Debugging"):
            results = self.evaluate()
            self.logger.info(f"Before Bug-fixing results = {results}")
            self.fix_bugs(bug_batch)
            results = self.evaluate()
            self.logger.info(f"After Bug-fixing results = {results}")
            self.timecode += 1
        return

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
            "Please Implement the `base_model_infer` method.")

    def check_debugger_args(self):
        raise NotImplementedError(
            "Please Implement the `check_debugger_args` method.")

    def bug_formatter(self, bug_batch):
        raise NotImplementedError(
            "Please Implement the `bug_formatter` method.")

    def get_dataloader(self, bug_data_args, formatted_bug_batch):
        raise NotImplementedError(
            "Please Implement the `get_dataloader` method.")

    def load_base_model(self, base_model_args):
        raise NotImplementedError(
            "Please Implement the `load_base_model` method.")

    def debugger_setup(self):
        raise NotImplementedError(
            "Please Implement the `debugger_setup` method.")

    def fix_bugs(self, bug_batch):
        raise NotImplementedError("Please Implement the `fix_bugs` method.")
