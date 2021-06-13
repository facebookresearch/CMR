
import torch
from transformers import BartTokenizer, BartConfig
import json


class OnlineDebuggingMethod():
    def __init__(self, logger=None):
        self.base_model = None
        self.debugger = None
        self.logger = logger
        self.use_cuda = torch.cuda.is_available()
        self.tokenizer = BartTokenizer.from_pretrained("bart-large")
        self.bug_loaders = []
        if self.use_cuda:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0
        return

    def load_bug_streams(self, bug_data_args):
        required_atts = ["bug_stream_json_path", "do_lowercase", "append_another_bos",
                         "max_input_length", "max_output_length", "task_name"]
        assert all([hasattr(bug_data_args, att) for att in required_atts])
        with open(bug_data_args.bug_stream_json_path) as f:
            bug_stream = json.load(f)
        self.bug_stream = bug_stream
        self.num_bug_batches = len(bug_stream)
        self.bug_batch_size = len(bug_stream[0])
        for bug_batch in self.bug_stream:
            formatted_bug_batch = self.bug_formatter(bug_batch)
            bug_dataloader = self.get_dataloader(
                bug_data_args, formatted_bug_batch)
            self.bug_loaders.append(bug_dataloader)
        return

    def evaluate(self):
        return

    def bug_formatter(self, bug_batch):
        raise NotImplementedError(
            "Please Implement the `bug_formatter` method.")

    def get_dataloader(self, bug_data_args, formatted_bug_batch):
        raise NotImplementedError(
            "Please Implement the `get_dataloader` method.")

    def load_base_model(self, base_model_path):
        raise NotImplementedError(
            "Please Implement the `load_base_model` method.")

    def debugger_setup(self):
        raise NotImplementedError(
            "Please Implement the `debugger_setup` method.")

    def fix_bugs(self, bug_batch, bug_fixing_args):
        raise NotImplementedError("Please Implement the `fix_bugs` method.")
