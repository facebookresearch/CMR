from .common_utils import OnlineDebuggingMethod
from semanticdebugger.models.mybart import MyBart
from semanticdebugger.models.utils import freeze_embeds, trim_batch, convert_model_to_single_gpu
import torch
import numpy as np
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup


class ContinualFinetuning(OnlineDebuggingMethod):
    def __init__(self, logger):
        super().__init__(logger=logger)

    def load_base_model(self, model_type, base_model_path):
        self.logger.info(
            f"Loading checkpoint from {base_model_path} for {model_type} .....")
        self.base_model = MyBart.from_pretrained(model_type,
                                                 state_dict=convert_model_to_single_gpu(torch.load(base_model_path)))
        self.logger.info(
            f"Loading checkpoint from {base_model_path} for {model_type} ..... Done!")
        if self.use_cuda:
            self.base_model.to(torch.device("cuda"))
            self.logger.info("Moving to the GPUs.")

    def debugger_setup(self, debugger_args):
        self.logger.info(f"Debugger Setup ......")
        self.logger.info(f"debugger_args: {debugger_args} ......")
        self.debugger_args = debugger_args
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.base_model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': debugger_args.weight_decay},
            {'params': [p for n, p in self.base_model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(self.optimizer_grouped_parameters,
                               lr=debugger_args.learning_rate, eps=debugger_args.adam_epsilon)

        # TODO: double check the decision about warup for fine-tuning
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=debugger_args.warmup_steps,
                                                         num_training_steps=debugger_args.total_steps)

        self.logger.info(f"Debugger Setup ...... Done!")

    def fix_bugs(self, bug_batch):
        self.base_model.train()
        if self.use_cuda:
            bug_batch = [b.to(torch.device("cuda")) for b in bug_batch]
