
import copy
import logging
import random
from semanticdebugger.debug_algs.cl_utils import _keep_first_answer
from semanticdebugger.models import run_bart
from semanticdebugger.task_manager.eval_metrics import evaluate_func
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

        if self.use_cuda:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0

        self.model_update_steps = 0  # number of updates over the base model.
        self.past_errors = []
        self.past_submissions = []
        return

    def _check_data_args(self, additional_args=[]):
        required_atts = ["submission_stream_data",
                         "upstream_eval_data",
                         "heldout_submission_data",
                         "do_lowercase",
                         "append_another_bos",
                         "max_input_length",
                         "max_output_length",
                         "task_name",
                         "num_beams",
                         "max_timecode"] + additional_args
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
                data_stream = json.load(f)
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

        wandb.log({"num_errors": len(formatted_bug_batch)}, step=self.timecode)
 
        wandb.log({"CER": len(self.past_errors) / len(self.past_submissions)}, step=self.timecode)

        result_dict["before_eval"] = _pack_as_dict(predictions, results, results_all)
        result_dict["error_ids"] = error_ids

        if return_raw_bug_examples:
            return formatted_bug_batch
        else:
            bug_train_loader, bug_eval_loader = self.get_dataloader(
                self.data_args, formatted_bug_batch, mode="both")
            return bug_train_loader, bug_eval_loader
  

    # The new evaluation pipeline.

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
            self.online_eval_results.append(result_dict) 
            if self.debugger_args.save_ckpt_freq > 0 and self.timecode % self.debugger_args.save_ckpt_freq == 0:
                self._save_base_model()
            self.logger.info("-"*50)
            self.timecode += 1

        #### Final evaluation ####
        self.final_evaluation()

        #### Save the final model ####
        self._save_base_model()

    def final_evaluation(self):
        self.logger.info("Start the final evaluation.")
        # TODO: 
        # self.overall_eval_results = {}
        # self.overall_eval_results["overall_oncoming_test"] = {key:
        #                                                       float(np.mean([r["before_eval"]["metric_results"][key]
        #                                                                      for r in self.online_eval_results]))
        #                                                       for key in self.metric.split("|")}
        # self.overall_eval_results["overall_error_number"] = len(self.past_errors)
        # self.overall_eval_results["overall_instant_fixing_rate"] = float(
        #     np.mean([r["instant_fixing_rate"] for r in self.online_eval_results]))
 
        # # Test the in-stream examples overall.
        # self.logger.info("Test final online KR.")
        # _, overall_instream_eval_dataloader = self.get_dataloader(
        #     self.data_args, self.past_submissions, mode="eval")
        # oie_predictions, oie_results, oie_results_all = self.evaluate(
        #     eval_dataloader=overall_instream_eval_dataloader, verbose=True)
        # self.overall_eval_results["final_OKR"] = oie_results

        # # Test the upstream forgetting eval.
        # self.logger.info("Test final upstream KR.")
        # oue_predictions, oue_results, oue_results_all = self.evaluate(
        #     eval_dataloader=self.upstream_eval_loader, verbose=True)
        # self.overall_eval_results["final_UKR"] = oue_results
        # self.logger.info("Finish the final evaluation.")

    
    def eval_knowledge_retention(self, result_dict):
        if self.timecode % self.debugger_args.kr_eval_freq != 0:
            return 
        
        ######################## UKR ######################## 
        self.logger.info(f"Start eval_knowledge_retention for UKR @ Timecode={self.timecode}")
        if self.debugger_args.kr_eval_mode == "loss":
            UKR_loss = self.evaluate(self.upstream_eval_loader, mode="loss")
        elif self.debugger_args.kr_eval_mode == "metric":
            predictions, results, results_all = self.evaluate(self.upstream_eval_loader)
            scores = results_all["EM"] 
            UKR = len([1 for s in scores if s == 1]) / len(scores)
        
        
        wandb.log({"UKR": UKR}, step=self.timecode)

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
        sampled_past_submissions = random.sample(self.past_submissions, k=self.debugger_args.okr_sample_size)
        _, past_submission_eval_loader = self.get_dataloader(self.data_args, sampled_past_submissions, mode="eval")
        self.logger.info(f"Start eval_knowledge_retention for OKR @ Timecode={self.timecode}")
        if self.debugger_args.kr_eval_mode == "loss":
            OKR = self.evaluate(past_submission_eval_loader, mode="loss")
        elif self.debugger_args.kr_eval_mode == "metric":
            predictions, results, results_all = self.evaluate(past_submission_eval_loader)
            scores = results_all["EM"] 
            OKR = len([1 for s in scores if s == 1]) / len(scores)
        self.logger.info(f"Online Knowledge Retation (OKR@{self.timecode}): {OKR:.4f}") 
        wandb.log({"OKR": OKR}, step=self.timecode)
         

    def eval_knowledge_generalization(self, result_dict):
        if self.timecode % self.debugger_args.kg_eval_freq != 0:
            return
        ######################## KG ######################## 
        self.logger.info(f"Start eval_knowledge_generalization for KG @ Timecode={self.timecode}")
        if self.debugger_args.kg_eval_mode == "loss":
            KG_loss = self.evaluate(self.heldout_submission_eval_loader, mode="loss")
        elif self.debugger_args.kg_eval_mode == "metric":
            predictions, results, results_all = self.evaluate(self.heldout_submission_eval_loader)
            scores = results_all["EM"] 
            KG = len([1 for s in scores if s == 1]) / len(scores) 
        wandb.log({"KG": KG}, step=self.timecode)
 
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
        wandb.log({"EFR": EFR}, step=self.timecode) 
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
