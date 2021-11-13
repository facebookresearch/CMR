from argparse import Namespace
from datetime import time
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


class OfflineCL(NoneCL):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "none_cl_offline_eval" 

    def _check_debugger_args(self):
        return  

    
    def online_debug(self):
        self.logger.info("Start Online Debugging with Dynamic Error Mode")
        self.logger.info(f"Number of Batches of Data: {self.num_data_batches}")
        self.logger.info(f"Data Batch Size: {self.data_batch_size};")
        self.timecode = 0
        
        
        # if self.debugger_args.save_ckpt_freq:
        #     # save the initial model as the 0-th model.
        #     self._save_base_model()
        data_args = self.data_args
        bug_eval_loaders = []
        for data_batch in tqdm(self.data_stream, desc="Creating the data loaders."): 
            data_batch += [item for item in data_batch if item["init_status"] == "error"]   # keep only the initial errors
            formatted_data_batch = self.data_formatter(data_batch) 
            _, eval_data_dataloader = self.get_dataloader(
                data_args, formatted_data_batch, mode="eval")
            bug_eval_loaders.append(eval_data_dataloader)
        
        for bug_eval_loader, data_eval_loader in tqdm(zip(bug_eval_loaders, self.data_eval_loaders), desc="Online Evaluation"):

            result_dict = {"timecode": self.timecode}   # start with 0
            
            if self.timecode+1 == len(self.data_eval_loaders):
                self.eval_knowledge_retention(result_dict)
                self.eval_knowledge_generalization(result_dict)

            # self._replay_based_eval(result_dict)
                
            
            _ = self._get_dynamic_errors(data_eval_loader, result_dict, return_raw_bug_examples=True) # we don't need the dataloader and empty cause false
            # bug_eval_loader = bug_eval_loaders[self.timecode]

            self.evaluate_error_fixing(result_dict, bug_eval_loader)
            self._update_result_dict(result_dict)
            # if self.debugger_args.save_ckpt_freq > 0 and self.timecode % self.debugger_args.save_ckpt_freq == 0:
            #     # self._save_base_model()
            #     self.save_result_file()
            self.logger.info("-"*50)
            self.timecode += 1

        #### Final evaluation ####
        self.final_evaluation()

        #### Save the final model ####
        self._save_base_model()