from argparse import Namespace
from logging import disable
from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
import numpy as np
import torch
from semanticdebugger.models.mybart import MyBart
from semanticdebugger.models import run_bart
from semanticdebugger.models.utils import (convert_model_to_single_gpu,
                                           freeze_embeds, trim_batch)
from semanticdebugger.task_manager.dataloader import GeneralDataset
from transformers import (AdamW, BartConfig, BartTokenizer,
                          get_linear_schedule_with_warmup)

from tqdm import tqdm


class NoneCL(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "none_cl" 

    def _check_debugger_args(self):
        return  

    def debugger_setup(self, debugger_args):
        self.logger.info(f"No debugger!")
        self.debugger_args = debugger_args
        self._check_debugger_args()
        return

    def fix_bugs(self, bug_loader, quiet=True):
        # bug_dataloader is from self.bug_loaders
        self.logger.info("No debugging at all.")
        return
