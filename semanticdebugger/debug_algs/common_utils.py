
import torch
from semanticdebugger.task_manager.dataloader import GeneralDataset
from transformers import BartTokenizer, BartConfig


class OnlineDebuggingMethod():
    def __init__(self, logger=None):
        self.base_model = None
        self.debugger = None
        self.logger = logger
        self.use_cuda = torch.cuda.is_available()
        self.tokenizer = BartTokenizer.from_pretrained("bart-large")

    def load_bug_streams(self, bug_stream, bug_data_args):
        required_atts = ["do_lowercase", "append_another_bos",
                         "max_input_length", "max_output_length", "task_name"]
        assert all([hasattr(bug_data_args, att) for att in required_atts])
        self.num_bug_batches = len(bug_stream)
        self.bug_batch_size = len(bug_stream[0])
        self.bug_stream_data = GeneralDataset(self.logger, bug_data_args, bug_data_args.train_file,
                                              data_type="train", is_training=True, task_name=bug_data_args.task_name)
        self.bug_stream_data.load_dataset(self.tokenizer, skip_cache=True)
        self.bug_stream_data.load_dataloader()

    def load_base_model(self, base_model_path):
        pass

    def debugger_setup(self):
        pass

    def evaluate(self):
        pass

    def fix_bugs(self, bug_batch):
        pass
