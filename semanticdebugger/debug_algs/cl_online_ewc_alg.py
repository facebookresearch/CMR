from argparse import Namespace
from logging import disable
import numpy as np
import torch
from semanticdebugger.models.mybart import MyBart
from semanticdebugger.models import run_bart
from semanticdebugger.models.utils import (convert_model_to_single_gpu,
                                           freeze_embeds, trim_batch)
from semanticdebugger.task_manager.dataloader import GeneralDataset
from transformers import (AdamW, BartConfig, BartTokenizer,
                          get_linear_schedule_with_warmup)

from semanticdebugger.debug_algs.commons import OnlineDebuggingMethod
from semanticdebugger.debug_algs.continual_finetune_alg import ContinualFinetuning
from tqdm import tqdm
from torch import nn
import torch
from torch.nn import functional as F
import abc

class OnlineEWC(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "online_ewc"

    def _check_debugger_args(self):
        super()._check_debugger_args()
        required_atts = [
            # ewc-related hyper parameters
            "ewc_lambda",
            "ewc_gamma", 
            "use_sampled_upstream"
            ]
        assert all([hasattr(self.debugger_args, att) for att in required_atts])
        

        return

    ### The same logic with the Simple Continual Fine-tuning Mehtod. ###
    def load_base_model(self, base_model_args):
        super().load_base_model(base_model_args)

    def base_model_infer(self, eval_dataloader, verbose=False):
        return super().base_model_infer(eval_dataloader, verbose)

    def data_formatter(self, bug_batch):
        return super().data_formatter(bug_batch)

    def get_dataloader(self, bug_data_args, formatted_bug_batch, mode="both"):
        return super().get_dataloader(bug_data_args, formatted_bug_batch, mode="both")

    ### END ###

    def debugger_setup(self, debugger_args):
        self.debugger_args = debugger_args
        self._check_debugger_args()
        self.logger.info(f"Debugger Setup ......")
        self.logger.info(f"debugger_args: {debugger_args} ......")

        # We don't set weight decay for them.
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.base_model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': debugger_args.weight_decay},
            {'params': [p for n, p in self.base_model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(self.optimizer_grouped_parameters,
                               lr=debugger_args.learning_rate, eps=debugger_args.adam_epsilon)

        # TODO: double check the decision about warm-up for fine-tuning
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=debugger_args.warmup_steps,
                                                         num_training_steps=debugger_args.total_steps)

        self.logger.info(f"Debugger Setup ...... Done!")

        # Initializing the EWC Regularzier.
        self.regularizer = EWCRegularizer()
        self.regularizer.online = True 
        self.regularizer.ewc_lambda = self.debugger_args.ewc_lambda
        self.regularizer.gamma = self.debugger_args.ewc_gamma
        self.regularizer.emp_FI = True # TODO: check later.
        self.regularizer.base_model = self.base_model

        return
        

    def fix_bugs(self, bug_loader, quiet=True):
        # bug_dataloader is from self.bug_loaders
        self.base_model.train()
        train_losses = []
        global_step = 0
        pad_token_id = self.tokenizer.pad_token_id

        #### For the first update ###
        if self.debugger_args.use_sampled_upstream and self.timecode==0:
            self.logger.info("Start the initial fisher info matrix computation....")
            upstream_dl, _ = self.get_dataloader(self.data_args, self.sampled_upstream_examples, mode="train")
            upstream_dl.args.train_batch_size = 1 
            upstream_fi_dl = upstream_dl.load_dataloader(do_return=True)
            self.regularizer.estimate_fisher(upstream_fi_dl, pad_token_id)
            self.logger.info("Start the initial fisher info matrix computation....Done!")


        for epoch_id in range(int(self.debugger_args.num_epochs)):
            for batch in tqdm(bug_loader.dataloader, desc=f"Bug-fixing Epoch {epoch_id}", disable=quiet):
                # here the batch is a mini batch of the current bug batch
                if self.use_cuda:
                    # print(type(batch[0]), batch[0])
                    batch = [b.to(torch.device("cuda")) for b in batch]
                batch[0], batch[1] = trim_batch(
                    batch[0], pad_token_id, batch[1])
                batch[2], batch[3] = trim_batch(
                    batch[2], pad_token_id, batch[3])
                # this is the task loss w/o any regularization
                loss = self.base_model(input_ids=batch[0], attention_mask=batch[1],
                                       decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                                       is_training=True)
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.

                ewc_loss = self.regularizer.ewc_loss()
                if self.regularizer.ewc_lambda>0:   # a hp to control the penalty weight.
                    # add the regularzation term.
                    loss = loss + self.regularizer.ewc_lambda * ewc_loss

                train_losses.append(loss.detach().cpu())
                loss.backward()

                if global_step % self.debugger_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.base_model.parameters(), self.debugger_args.max_grad_norm)
                    self.optimizer.step()    # We have accumulated enough gradients
                    self.scheduler.step()
                    self.base_model.zero_grad()

        # TODO: build bsz=1 dataloader for update the fisher information matrix
        fisher_dataloader = bug_loader  # can we copy this object?
        fisher_dataloader.args.train_batch_size = 1 
        fi_dl = fisher_dataloader.load_dataloader(do_return=True)
        self.regularizer.estimate_fisher(fi_dl, pad_token_id)
        return


class EWCRegularizer(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier.
    '''

    def __init__(self, ):
        super().__init__()
        self.base_model = None # the bart model or other possible models to
        # -EWC:
        # -> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.ewc_lambda = 0
        # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.gamma = 1.
        # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.online = True
        # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.fisher_n = None
        # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.emp_FI = True # otherwise we need to do the inference decoding.
        # -> keeps track of number of quadratic loss terms (for "offline EWC")
        self.EWC_task_count = 0

    def estimate_fisher(self, data_loader, pad_token_id):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.

        [data_loader]: <DataSet> to be used to estimate FI-matrix; give batches of size 1
        '''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.base_model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.base_model.training
        self.base_model.eval()

        # Create data-loader to give batches of size 1
        # data_loader = utils.get_data_loader(
        #     dataset, batch_size=1, cuda=self._is_on_cuda(), collate_fn=collate_fn)
        
        # TODO: why batch size =1 ?
        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index, batch in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            # x = x.to(self.base_model._device())
            batch = [b.to(torch.device("cuda")) for b in batch]
            batch[0], batch[1] = trim_batch(
                    batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(
                batch[2], pad_token_id, batch[3])
 
            # output = self.base_model(x) 
            assert self.emp_FI
            # -use provided label to calculate loglikelihood --> "empirical Fisher":
            # label = torch.LongTensor([y]) if type(y) == int else y
            # label = label.to(self.base_model._device())
            # calculate negative log-likelihood
            # negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            nll_loss = self.base_model(input_ids=batch[0], attention_mask=batch[1],
                                       decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                                       is_training=True)
            # Calculate gradient of negative loglikelihood
            self.base_model.zero_grad()
            nll_loss.backward() 
            ###


            # Square gradients and keep running sum
            for n, p in self.base_model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.base_model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count+1), p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count+1), est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.base_model.train(mode=mode)

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count > 0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count+1):
                for n, p in self.base_model.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(
                            n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(
                            n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma*fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=torch.device("cuda"))
