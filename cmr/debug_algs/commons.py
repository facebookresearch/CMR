# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import copy
import logging
import random
from cmr.debug_algs.cl_utils import _keep_first_answer
from cmr.models import run_bart
from cmr.task_manager.eval_metrics import evaluate_func
import torch
from transformers import BartTokenizer, BartConfig
import json
from tqdm import tqdm
import os
import numpy as np
import wandb


def _pack_as_dict(predictions, results, results_all):
    return {"predictions": predictions, "metric_results": results, "metric_results_detailed": results_all}


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

        self.submission_eval_loaders = []   # for online dynamic streams
        self.upstream_eval_loader = None    # for UKR
        self.heldout_submission_eval_loader = None  # for KG eval
 

        # utils
        self.use_cuda = torch.cuda.is_available()
        self.tokenizer = BartTokenizer.from_pretrained("bart-large")
        self.timecode = None
        self.metric = "EM|QA-F1"

        # for dynamic stream mode
        self.data_eval_loaders = []
        self.online_eval_results = []
        self.last_OKR = None; self.last_UKR = None; self.last_KG = None 

        if self.use_cuda:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0

        self.model_update_steps = 0  # number of updates over the base model.
        self.past_errors = []
        self.past_submissions = []
        
        return


    def save_result_file(self):
        output_info = {}
        output_info["method_class"] = self.name
        output_info["base_model_args"] = str(self.base_model_args)
        output_info["debugger_args"] = str(self.debugger_args)
        output_info["data_args"] = str(self.data_args)
        output_info["model_update_steps"] = self.model_update_steps
        output_info["online_eval_results"] = self.online_eval_results 
        
        # if args.cl_method_name in ["offline_debug"]:
        #     output_info["offline_bound_results"] = offline_bound_results
        #     logger.info(f"eval_results_overall_bug: {offline_bound_results['eval_results_overall_bug']['metric_results']}")
        #     logger.info(f"eval_results_overall_forget: {offline_bound_results['eval_results_overall_forget']['metric_results']}")
        with open(self.data_args.result_file, "w") as f:
            json.dump(output_info, f)
        self.logger.info(f"Updated result file: {self.data_args.result_file} at Timecode: {self.timecode}.")

    def _check_data_args(self, additional_args=[]):
        required_atts = ["submission_stream_data",
                         "stream_id",
                         "upstream_eval_data",
                         "heldout_submission_data",
                         "do_lowercase",
                         "append_another_bos",
                         "max_input_length",
                         "max_output_length",
                         "task_name",
                         "num_beams",
                         "max_timecode",
                         "result_file"] + additional_args
        assert all([hasattr(self.data_args, att) for att in required_atts])
        return

    def load_data(self, data_args, given_data_stream=None):
        """"For loading the data stream for dynamic building the errors."""
        self.data_args = data_args
        self._check_data_args()  # additional_args=["data_stream_json_path", "accumulate_eval_freq"]
        # Load bug stream
        if given_data_stream:
            data_stream = given_data_stream
        else:
            with open(data_args.submission_stream_data) as f:
                data_stream = json.load(f)[data_args.stream_id]
                self.logger.info(f"Loading the stream from {f.name} and use the ${data_args.stream_id} part.")
        self.data_stream = data_stream
        self.num_data_batches = len(data_stream)
        self.data_batch_size = len(data_stream[0])
        # Create data loaders for each error batch.
        all_formatted_data = []
        self.data_eval_loaders = []
        self.online_eval_results = []
        for data_batch in tqdm(self.data_stream, desc="Creating the data loaders."):
            if data_args.max_timecode > 0 and len(self.data_eval_loaders) >= data_args.max_timecode:
                break
            formatted_data_batch = self.data_formatter(data_batch)
            all_formatted_data += formatted_data_batch
            _, eval_data_dataloader = self.get_dataloader(
                data_args, formatted_data_batch, mode="eval")
            self.data_eval_loaders.append(eval_data_dataloader)

        self.all_formatted_data = all_formatted_data
 
        # Create loaders for the sampled pass examples for evaluation.
        with open(data_args.upstream_eval_data) as f:
            upstream_eval_examples = [json.loads(line) for line in f.read().splitlines()]
        upstream_eval_examples = self.data_formatter(upstream_eval_examples)
        self.logger.info(f"load_data: len(upstream_eval_examples)={len(upstream_eval_examples)}")
        _, self.upstream_eval_loader = self.get_dataloader(
            data_args, upstream_eval_examples, mode="eval")

        # Create loaders for the sampled pass examples for evaluation.
        with open(data_args.heldout_submission_data) as f:
            heldout_eval_examples = [json.loads(line) for line in f.read().splitlines()]
        heldout_eval_examples = self.data_formatter(heldout_eval_examples)
        self.logger.info(f"load_data: len(heldout_eval_examples)={len(heldout_eval_examples)}")
        _, self.heldout_submission_eval_loader = self.get_dataloader(
            data_args, heldout_eval_examples, mode="eval")

        

    def _get_dynamic_errors(self, data_eval_loader, result_dict, return_raw_bug_examples=False):
        ############### Get the errors dynamically. ###############
        self.logger.info(
            f"Evaluating to get errors .... Timecode: {self.timecode}")
        self.past_submissions += data_eval_loader.data
        predictions, results, results_all = self.evaluate(data_eval_loader)
        self.logger.info(f"Before Error Fixing: {results}")

        # self.logger.info(
        #     f"Doing-Nothing Instant EM: {self.instant_doing_nothing_EM[self.timecode]}")

        ### Pack the error examples for training. ###
        errors = []
        error_ids = []
        for (_input, _truth, _id), prediction, em, f1 in zip(data_eval_loader.data,
                                                             predictions,
                                                             results_all["EM"],
                                                             results_all["QA-F1"]):
            # self.logger.info(f"{example}")
            # self.logger.info(f"{prediction}")
            # self.logger.info(f"{em}")
            if em == 0:  # TODO: this is the condition to judge if it is a bug.
                bug = {}
                bug["id"] = _id
                bug["input"] = _input
                bug["truth"] = _truth
                bug["mistake"] = prediction
                errors.append(bug)
                error_ids.append(_id)
                self.past_errors.append(bug)
        formatted_bug_batch = self.data_formatter(errors)
        self.logger.info(f"Found {len(formatted_bug_batch)} errors.")
        
        SR = 1 - len(error_ids)/len(predictions)
        CSR = 1 - len(self.past_errors) / len(self.past_submissions)
        wandb.log({"num_errors": len(formatted_bug_batch)}, step=self.timecode)
        wandb.log({"CSR": CSR}, step=self.timecode)
        wandb.log({"SR": SR}, step=self.timecode)

        result_dict["before_eval_results"] = _pack_as_dict(predictions, results, results_all)
        result_dict["before_error_ids"] = error_ids        
        result_dict["SR"] = SR
        result_dict["CSR"] = CSR
        
        if return_raw_bug_examples:
            return formatted_bug_batch
        else:
            bug_train_loader, bug_eval_loader = self.get_dataloader(
                self.data_args, formatted_bug_batch, mode="both")
            return bug_train_loader, bug_eval_loader
  
    def _update_result_dict(self, result_dict):
        # if self.last_OKR is None or self.last_KG is None or self.last_UKR is None:
        #     pass
        # else:

        scores = [result_dict.get("CSR", 0.0), result_dict.get("EFR", 0.0)]
        if self.last_OKR:
            scores.append(self.last_OKR)
            scores.append(self.last_UKR)
            scores.append(self.last_KG)
        result_dict["Overall"] = float(np.mean(scores))
        wandb.log({"Overall": result_dict["Overall"]}, step=self.timecode)
        self.logger.info(f'Overall: {result_dict["Overall"]} from scores={scores}')
        self.online_eval_results.append(result_dict)

    def online_debug(self):
        self.logger.info("Start Online Debugging with Dynamic Error Mode")
        self.logger.info(f"Number of Batches of Data: {self.num_data_batches}")
        self.logger.info(f"Data Batch Size: {self.data_batch_size};")
        self.timecode = 0
        
        
        if self.debugger_args.save_ckpt_freq:
            # save the initial model as the 0-th model.
            self._save_base_model()

        for data_eval_loader in tqdm(self.data_eval_loaders, desc="Online Debugging"):

            result_dict = {"timecode": self.timecode}   # start with 0
            
            self.eval_knowledge_retention(result_dict)
            self.eval_knowledge_generalization(result_dict)

            # self._replay_based_eval(result_dict)
            
            bug_train_loader, bug_eval_loader = self._get_dynamic_errors(data_eval_loader, result_dict)

            ############### CORE ###############
            # Fix the bugs by mini-batch based "training"
            self.logger.info(f"Start error-fixing .... Timecode: {self.timecode}")
            self.fix_bugs(bug_train_loader)   # for debugging
            self.logger.info("Start error-fixing .... Done!")
            ############### CORE ###############

            self.evaluate_error_fixing(result_dict, bug_eval_loader)
            self._update_result_dict(result_dict)
            

            if self.debugger_args.save_ckpt_freq > 0 and self.timecode % self.debugger_args.save_ckpt_freq == 0:
                self._save_base_model()
                self.save_result_file()
            self.logger.info("-"*50)
            self.timecode += 1

        #### Final evaluation ####
        self.final_evaluation()

        #### Save the final model ####
        self._save_base_model()

    def final_evaluation(self):
        self.logger.info("Start the final evaluation.")
        # TODO: 
        self.logger.info("Nothing here.")

    
    def eval_knowledge_retention(self, result_dict):
        if self.timecode == self.data_args.max_timecode-1:
            pass
        elif self.timecode % self.debugger_args.kr_eval_freq == 0:
            pass
        else:
            return 
        
        ######################## UKR ######################## 
        self.logger.info(f"Start eval_knowledge_retention for UKR @ Timecode={self.timecode}")
        if self.debugger_args.kr_eval_mode == "loss":
            UKR_loss = self.evaluate(self.upstream_eval_loader, mode="loss")
        elif self.debugger_args.kr_eval_mode == "metric":
            predictions, results, results_all = self.evaluate(self.upstream_eval_loader)
            scores = results_all["EM"] 
            UKR = len([1 for s in scores if s == 1]) / len(scores)
        
        result_dict["UKR"] = UKR
        wandb.log({"UKR": UKR}, step=self.timecode)
        self.last_UKR = UKR

        # UKR_loss = self.evaluate(self.upstream_eval_loader, mode="loss")
        # wandb.log({"UKR_loss": UKR_loss}, step=self.timecode)
        self.logger.info(f"Upstream Knowledge Retation (UKR@{self.timecode}): {UKR:.4f}") 

        ######################## OKR ######################## 
        if not self.past_submissions:
            return 
        rng = random.Random(self.debugger_args.okr_sample_seed) # fixed for different methods e.g., 1337
        if len(self.past_submissions) < self.debugger_args.okr_sample_size:
            self.logger.info(f"len(self.past_submissions) = {len(self.past_submissions)} \
                < self.debugger_args.okr_sample_size = {self.debugger_args.okr_sample_size}")
            return 
        sampled_past_submissions = rng.sample(self.past_submissions, k=self.debugger_args.okr_sample_size)
        result_dict["OKR_sampled_ids"] = [_id for _input, _truth, _id in sampled_past_submissions]
        result_dict["OKR_sampled_ids"].sort()
        _, past_submission_eval_loader = self.get_dataloader(self.data_args, sampled_past_submissions, mode="eval")
        self.logger.info(f"Start eval_knowledge_retention for OKR @ Timecode={self.timecode}")
        if self.debugger_args.kr_eval_mode == "loss":
            OKR = self.evaluate(past_submission_eval_loader, mode="loss")
        elif self.debugger_args.kr_eval_mode == "metric":
            predictions, results, results_all = self.evaluate(past_submission_eval_loader)
            scores = results_all["EM"] 
            OKR = len([1 for s in scores if s == 1]) / len(scores)
        self.logger.info(f"Online Knowledge Retation (OKR@{self.timecode}): {OKR:.4f}") 
        result_dict["OKR"] = OKR
        self.last_OKR = OKR
        wandb.log({"OKR": OKR}, step=self.timecode)
        
         

    def eval_knowledge_generalization(self, result_dict):
        if self.timecode == self.data_args.max_timecode-1:
            pass
        elif self.timecode % self.debugger_args.kg_eval_freq == 0:
            pass
        else:
            return 
        ######################## KG ######################## 
        self.logger.info(f"Start eval_knowledge_generalization for KG @ Timecode={self.timecode}")
        if self.debugger_args.kg_eval_mode == "loss":
            KG_loss = self.evaluate(self.heldout_submission_eval_loader, mode="loss")
        elif self.debugger_args.kg_eval_mode == "metric":
             # TODO: get a decomposed version?
            predictions, results, results_all = self.evaluate(self.heldout_submission_eval_loader)
            scores = results_all["EM"] 
            KG = len([1 for s in scores if s == 1]) / len(scores) 
        result_dict["KG"] = KG         
        wandb.log({"KG": KG}, step=self.timecode) 
        self.last_KG = KG
        self.logger.info(f"Future Knowledge Generalization (KG@{self.timecode}): {KG:.4f}")         
        
    def evaluate_error_fixing(self, result_dict, bug_eval_loader):
        after_predictions, after_results, after_results_all = self.evaluate(bug_eval_loader)
        fixed_ids = []
        unfixed_ids = []
        for (_input, _truth, _id),  score_after in zip(bug_eval_loader.data, after_results_all["EM"]): 
            if score_after == 1:
                fixed_ids.append(_id)
            else:
                unfixed_ids.append(_id)
        EFR = len(fixed_ids) / len(fixed_ids+unfixed_ids)
        result_dict["EFR"] = EFR
        wandb.log({"EFR": EFR}, step=self.timecode) 
        self.logger.info(f"EFR={EFR}")
        return EFR
      # So the 0-th checkpoint should be the original base model.

    def _save_base_model(self, ckpt_name=None):
        output_dir = self.debugger_args.ckpt_dir
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

    def evaluate(self, eval_dataloader=None, verbose=False, mode="metric"):
        """Evaluates the performance"""
        if not eval_dataloader:
            self.logger.info("evaluate with submission eval loaders")
            eval_dataloader = self.submission_eval_loaders[self.timecode]

        if mode == "metric":
            predictions = self.base_model_infer(eval_dataloader, verbose)
            assert len(predictions) == len(eval_dataloader)
            predictions = [p.strip() for p in predictions]
            results, results_all = evaluate_func(
                predictions, eval_dataloader.data, self.metric, return_all=True)
            return predictions, results, results_all
        elif mode == "loss":
            examples = eval_dataloader.data
            _examples = _keep_first_answer(examples)
            tmp_data_args = copy.deepcopy(self.data_args)
            tmp_data_args.predict_batch_size = 8    # TODO: set an arg. 
            eval_loader, _ = self.get_dataloader(tmp_data_args, _examples, mode="train", is_training=False) # fix of the order
            losses = run_bart.inference(
                self.base_model, eval_loader, compute_loss=True, loss_only=True, logger=self.logger)
            mean_loss = sum(losses) / len(examples)
            return mean_loss
            

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
