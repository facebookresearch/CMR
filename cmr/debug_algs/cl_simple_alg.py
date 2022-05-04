# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from argparse import Namespace
from logging import disable
import numpy as np
import torch
from cmr.models.mybart import MyBart
from cmr.models import run_bart
from cmr.models.utils import (convert_model_to_single_gpu,
                                           freeze_embeds, trim_batch)
from cmr.task_manager.dataloader import GeneralDataset
from transformers import (AdamW, BartConfig, BartTokenizer,
                          get_linear_schedule_with_warmup)

from cmr.debug_algs.commons import OnlineDebuggingMethod
from tqdm import tqdm
import copy


class ContinualFinetuning(OnlineDebuggingMethod):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "simple_cl"
        

    def _check_debugger_args(self):
        required_atts = ["weight_decay",
                         "learning_rate",
                         "adam_epsilon",
                         "warmup_steps",
                         "total_steps",
                         "num_epochs",
                         "gradient_accumulation_steps",
                         "max_grad_norm",
                         "diff_loss_weight"]
        assert all([hasattr(self.debugger_args, att) for att in required_atts])
        return

    def load_base_model(self, base_model_args, mode="online_debug"):
        self.base_model_args = base_model_args
        
        model_type, base_model_path = base_model_args.model_type, base_model_args.base_model_path
        self.logger.info(
            f"Loading checkpoint from {base_model_path} for {model_type} .....")
        self.base_model = MyBart.from_pretrained(model_type,
                                                 state_dict=convert_model_to_single_gpu(torch.load(base_model_path)))
        self.logger.info(
            f"Loading checkpoint from {base_model_path} for {model_type} ..... Done!")
        if self.use_cuda:
            self.base_model.to(torch.device("cuda"))
            self.logger.info("Moving to the GPUs.")
            if self.n_gpu > 1:
                self.base_model = torch.nn.DataParallel(self.base_model)

    def base_model_infer(self, eval_dataloader, verbose=False):
        self.base_model.eval()
        model = self.base_model if self.n_gpu == 1 else self.base_model.module
        predictions = run_bart.inference(model, eval_dataloader, save_predictions=False, verbose=verbose,
                                         logger=self.logger, return_all=False, predictions_only=True, args=Namespace(quiet=True))
        return predictions

    def data_formatter(self, bug_batch):
        # The continual fine-tuning method only uses the correct answers for fixing bugs.
        formatted_bug_batch = []
        for bug in bug_batch:
            # if "id" not in bug:
            #     _id = len(formatted_bug_batch)
            _id = bug["id"]
            _input = bug["input"]
            # _mistake = bug["mistake"]

            # TODO: only for now debugging.
            if "truth" in bug:
                _truth = bug["truth"]   # a list of answers
            else:
                _truth = bug["output"]   # a list of answers

            formatted_bug_batch.append((_input, _truth, _id))
        return formatted_bug_batch

    def get_dataloader(self, bug_data_args, formatted_bug_batch, mode="both", is_training="self"):
        # mini bug-batch size.
        assert hasattr(bug_data_args, "train_batch_size")
        assert hasattr(bug_data_args, "predict_batch_size")
        train_bug_dataloader, eval_bug_dataloader = None, None
        if mode == "both" or mode == "train":
            # for error-fixing
            train_bug_dataloader = GeneralDataset(self.logger, bug_data_args, None,
                                                  data_type="train", is_training=True,
                                                  task_name=bug_data_args.task_name,
                                                  given_data=formatted_bug_batch)
            train_bug_dataloader.load_dataset(
                self.tokenizer, skip_cache=True, quiet=True)
            train_bug_dataloader.load_dataloader(is_training=is_training)
        if mode == "both" or mode == "eval":
            # for evaluation
            eval_bug_dataloader = GeneralDataset(self.logger, bug_data_args, None,
                                                 data_type="dev", is_training=False,
                                                 task_name=bug_data_args.task_name,
                                                 given_data=formatted_bug_batch)
            eval_bug_dataloader.load_dataset(
                self.tokenizer, skip_cache=True, quiet=True)
            eval_bug_dataloader.load_dataloader()

        return train_bug_dataloader, eval_bug_dataloader

 
    def reset_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.base_model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.debugger_args.weight_decay},
            {'params': [p for n, p in self.base_model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(self.optimizer_grouped_parameters,
                               lr=self.debugger_args.learning_rate, eps=self.debugger_args.adam_epsilon)

        # TODO: double check the decision about warm up for fine-tuning
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.debugger_args.warmup_steps,
                                                         num_training_steps=self.debugger_args.total_steps)
        self.logger.info(f"optimizer & scheduler Setup ...... Done!")

    def debugger_setup(self, debugger_args):
        self.debugger_args = debugger_args
        self._check_debugger_args()
        self.logger.info(f"Debugger Setup ......")
        self.logger.info(f"debugger_args: {debugger_args} ......")
        self.reset_optimizer()
        
        self.logger.info(f"Debugger Setup ...... Done!")
        return

    def fix_bugs(self, bug_loader, quiet=True):
        # bug_dataloader is from self.bug_loaders
        self.base_model.train()
        train_losses = []
        global_step = 0
        if self.debugger_args.diff_loss_weight > 0:
            last_weights = copy.deepcopy(list(self.base_model.parameters()))
        for epoch_id in range(int(self.debugger_args.num_epochs)):
            for batch in tqdm(bug_loader.dataloader, desc=f"Bug-fixing Epoch {epoch_id}", disable=quiet):
                global_step += 1
                # here the batch is a mini batch of the current bug batch
                if self.use_cuda:
                    # print(type(batch[0]), batch[0])
                    batch = [b.to(torch.device("cuda")) for b in batch]
                pad_token_id = self.tokenizer.pad_token_id
                batch[0], batch[1] = trim_batch(
                    batch[0], pad_token_id, batch[1])
                batch[2], batch[3] = trim_batch(
                    batch[2], pad_token_id, batch[3])
                loss = self.base_model(input_ids=batch[0], attention_mask=batch[1],
                                       decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                                       is_training=True)
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                
                # For L2 norm 

                if self.debugger_args.diff_loss_weight > 0:
                    diff_loss = torch.Tensor([0]).to("cuda" if torch.cuda.is_available() else "cpu")
                    # Iterate over base_weights and curr_weights and accumulate the euclidean norm
                    # of their differences
                    
                    curr_weights = list(self.base_model.parameters())
                    for base_param, curr_param in zip(last_weights, curr_weights):
                        diff_loss += (curr_param - base_param).pow(2).sum()
                    # self.logger.info(f"loss={loss}; diff_loss={diff_loss}; l2w={self.debugger_args.diff_loss_weight}")
                    loss = loss + self.debugger_args.diff_loss_weight * diff_loss


                train_losses.append(loss.detach().cpu())
                loss.backward()
                self.model_update_steps += 1

                if global_step % self.debugger_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.base_model.parameters(), self.debugger_args.max_grad_norm)
                    self.optimizer.step()    # We have accumulated enough gradients
                    self.scheduler.step()
                    self.base_model.zero_grad()
                    # last_weights = copy.deepcopy(list(self.base_model.parameters())) # update the last weights
        if self.debugger_args.diff_loss_weight > 0:
            del last_weights
        return
