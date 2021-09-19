
import logging
import random
from semanticdebugger.task_manager.eval_metrics import evaluate_func
import torch
from transformers import BartTokenizer, BartConfig
import json
from tqdm import tqdm
import os
import numpy as np


def _pack_as_dict(predictions, results, results_all):
    return {"predictions": predictions, "metric_results": results, "metric_results_detailed": results_all}


class OnlineDebuggingMethod():
    def __init__(self, logger=None):
        self.name = "base_class"
        self.stream_mode = "static"
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
        self.bug_stream = []
        self.bug_train_loaders = []
        self.bug_eval_loaders = []
        self.bug_all_train_loader = None
        self.bug_all_eval_loader = None
        self.replay_eval_loaders = []

        self.sampled_passes = []
        self.forget_eval_loader = None

        self.all_bug_examples = []
        self.sampled_upstream_examples = []

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
        return

    def _check_data_args(self, additional_args=[]):
        required_atts = ["bug_stream_json_path",
                         "pass_pool_jsonl_path",
                         "use_sampled_upstream",
                         "sampled_upstream_json_path",
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
        self._check_data_args(
            additional_args=["data_stream_json_path", "accumulate_eval_freq"])
        # Load bug stream
        if given_data_stream:
            data_stream = given_data_stream
        else:
            with open(data_args.data_stream_json_path) as f:
                data_stream = json.load(f)
        self.data_stream = data_stream
        self.num_data_batches = len(data_stream)
        self.data_batch_size = len(data_stream[0])
        # Create data loaders for each error batch.
        all_formatted_data = []
        # accumulate_doing_nothing_EM = []
        # instant_doing_nothing_EM = []
        # self.all_initial_pass_ids = set()
        # self.all_initial_error_ids = set()
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
            #
            # doing_nothing_EM = float(
            #     np.mean([item["score"]["EM"] for item in data_batch]))
            # instant_doing_nothing_EM.append(doing_nothing_EM)
            # accumulate_doing_nothing_EM.append(
            #     float(np.mean(instant_doing_nothing_EM)))
            # for item in data_batch:
            #     if item["init_status"] == "pass":
            #         self.all_initial_pass_ids.add(item["id"])
            #     else:
            #         self.all_initial_error_ids.add(item["id"])

        # Create the list of replay eval loaders for monitoring the performance.
        if data_args.replay_stream_json_path:
            with open(data_args.replay_stream_json_path) as f:
                replay_stream = json.load(f)
            self.replay_stream = replay_stream
            for replay_batch in tqdm(self.replay_stream, desc="Creating the replay data loaders."):
                formatted_replay_batch = self.data_formatter(replay_batch)
                _, eval_replay_dataloader = self.get_dataloader(
                    data_args, formatted_replay_batch, mode="eval")
                self.replay_eval_loaders.append(eval_replay_dataloader)
            self.replay_eval_loaders = self.replay_eval_loaders[:self.num_bug_batches]
        else:
            self.logger.info("Not creating the replay-stream for evaluation.")
        # assert len(self.replay_eval_loaders) == self.num_bug_batches

        # Init other useful statistics.
        
        # self.instant_doing_nothing_EM = instant_doing_nothing_EM
        # self.accumulate_doing_nothing_EM = accumulate_doing_nothing_EM
        self.all_formatted_data = all_formatted_data

        if data_args.pass_pool_jsonl_path: 
            # Create loaders for the sampled pass examples for evaluation.
            with open(data_args.pass_pool_jsonl_path) as f:
                pass_examples = [json.loads(line) for line in set(f.read().splitlines())]
            self.sampled_passes = pass_examples
            pass_examples = self.data_formatter(pass_examples)
            _, self.forget_eval_loader = self.get_dataloader(
                data_args, pass_examples, mode="eval")
        
        self.sampled_upstream_examples = None 
        if data_args.use_sampled_upstream and data_args.sampled_upstream_json_path:
            # Create loaders for the sampled pass examples as the initial memory.
                # for the ER/MIR use
                # if for other memory-based methon, this part of data should be encoded offline
            with open(data_args.sampled_upstream_json_path) as f:
                sampled_upstream_examples = [json.loads(line)
                                                for line in set(f.read().splitlines())]
            self.sampled_upstream_examples = self.upstream_data_formatter(
                sampled_upstream_examples)

    def _replay_based_eval(self, result_dict):
        if not self.data_args.replay_stream_json_path:
            self.logger.info("Not doing replay-based eval.")
            return
        replay_eval_loader = self.replay_eval_loaders[self.timecode]
        self.logger.info(
            f"Evaluating the replayed data from the recent episodes.... Timecode: {self.timecode}")
        predictions, results, results_all = self.evaluate(replay_eval_loader)
        result_dict["replay_eval"] = _pack_as_dict(predictions, results, results_all)

        base_EM = float(
            np.mean([i["init_status"] == "pass" for i in self.replay_stream[self.timecode]]))
        # result_dict["model0_replay_eval"] = {"EM": base_EM}

        self.logger.info(f"Replayed Performance: {results['EM']} vs Model_0: {base_EM}")

    def _get_dynamic_errors(self, data_eval_loader, result_dict):
        ############### Get the errors dynamically. ###############
        self.logger.info(
            f"Evaluating to get errors .... Timecode: {self.timecode}")
        predictions, results, results_all = self.evaluate(data_eval_loader)

        self.logger.info(f"Before Error Fixing: {results['EM']}")
        # self.logger.info(
        #     f"Doing-Nothing Instant EM: {self.instant_doing_nothing_EM[self.timecode]}")

        ### Pack the error examples for training. ###
        errors = []
        error_ids = []
        for (_input, _truth, _id), prediction, em in zip(data_eval_loader.data, predictions, results_all["EM"]):
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
                self.overall_errors.append(bug)
        formatted_bug_batch = self.data_formatter(errors)
        bug_train_loader, _ = self.get_dataloader(
            self.data_args, formatted_bug_batch, mode="train")

        self.logger.info(f"Found {len(errors)} errors.")
        result_dict["before_eval"] = _pack_as_dict(predictions, results, results_all)
        result_dict["error_ids"] = error_ids
        return bug_train_loader

    def _log_episode_result(self, result_dict, data_eval_loader):
        self.logger.info(
            f"Evaluating again to analyze the performance .... Timecode: {self.timecode}")
        after_predictions, after_results, after_results_all = self.evaluate(
            data_eval_loader)
        result_dict["after_eval"] = _pack_as_dict(
            after_predictions, after_results, after_results_all)
        
        self.logger.info(f"After Error Fixing: {after_results['EM']}")

        forgotten_examples, retained_ids, fixed_ids, unfixed_ids = self._eval_forget_and_fix(
            data_eval_loader.data, result_dict["before_eval"], result_dict["after_eval"])
        instant_fixing_rate = len(fixed_ids) / (len(fixed_ids) + len(unfixed_ids))
        instant_retention_rate = len(retained_ids)/(len(retained_ids) + len(forgotten_examples))
        self.logger.info(f"Instant Fixing Rate: {instant_fixing_rate}")
        self.logger.info(f"Instant Retention Rate: {instant_retention_rate}")

        self.logger.info("-"*50)
        # Start the logging.

        
        result_dict["forgotten_examples"] = forgotten_examples
        result_dict["retained_ids"] = retained_ids
        result_dict["fixed_ids"] = fixed_ids
        result_dict["unfixed_ids"] = unfixed_ids
        result_dict["instant_fixing_rate"] = instant_fixing_rate
        result_dict["instant_retention_rate"] = instant_retention_rate
        # result_dict["model0_instant_EM"] = self.instant_doing_nothing_EM[self.timecode]
        self.seen_stream_data += data_eval_loader.data
        # result_dict["doing-nothing_accmulative_EM"] = self.accumulate_doing_nothing_EM[self.timecode]

        self.online_eval_results.append(result_dict)

    # The new evaluation pipeline.
    def online_debug(self):
        self.logger.info("Start Online Debugging with Dynamic Error Mode")
        self.logger.info(f"Number of Batches of Data: {self.num_data_batches}")
        self.logger.info(f"Data Batch Size: {self.data_batch_size};")
        self.timecode = 0

        if self.debugger_args.save_all_ckpts:
            # save the initial model as the 0-th model.
            self._save_base_model()

        self.overall_errors = []
        self.seen_stream_data = []
        for data_eval_loader in tqdm(self.data_eval_loaders, desc="Online Debugging (Dynamic)"):

            result_dict = {"timecode": self.timecode}   # start with 0

            self._replay_based_eval(result_dict)
            bug_train_loader = self._get_dynamic_errors(data_eval_loader, result_dict)

            ############### CORE ###############
            # Fix the bugs by mini-batch based "training"
            self.logger.info(f"Start bug-fixing .... Timecode: {self.timecode}")
            self.fix_bugs(bug_train_loader)   # for debugging
            self.logger.info("Start bug-fixing .... Done!")
            ############### CORE ###############

            self._log_episode_result(result_dict, data_eval_loader)
            self.timecode += 1

            if self.debugger_args.save_all_ckpts:
                self._save_base_model()

        #### Final evaluation ####
        self.final_evaluation()

    def final_evaluation(self):
        self.logger.info("Start the final evaluation.")

        self.overall_eval_results = {}
        self.overall_eval_results["overall_oncoming_test"] = {key:
                                                              float(np.mean([r["before_eval"]["metric_results"][key]
                                                                             for r in self.online_eval_results]))
                                                              for key in self.metric.split("|")}
        if self.data_args.replay_stream_json_path:
            self.overall_eval_results["overall_replay_test"] = {key:
                                                                float(np.mean([r["replay_eval"]["metric_results"][key]
                                                                               for r in self.online_eval_results]))
                                                                for key in self.metric.split("|")}
            # self.overall_eval_results["model0_replay_test"] = {"EM": float(
            #     np.mean([r["model0_replay_eval"]["EM"] for r in self.online_eval_results]))}

        self.overall_eval_results["overall_error_number"] = len(self.overall_errors)
        self.overall_eval_results["overall_instant_fixing_rate"] = float(
            np.mean([r["instant_fixing_rate"] for r in self.online_eval_results]))

        # Re-test the past errors.
        self.logger.info("Re-test the past errors.")
        _, overall_error_eval_dataloader = self.get_dataloader(
            self.data_args, self.data_formatter(self.overall_errors), mode="eval")
        oev_predictions, oev_results, oev_results_all = self.evaluate(
            eval_dataloader=overall_error_eval_dataloader, verbose=True)
        self.overall_eval_results["final_fixing_rate"] = oev_results

        # Test the in-stream examples overall.
        self.logger.info("Test the in-stream examples overall.")
        _, overall_instream_eval_dataloader = self.get_dataloader(
            self.data_args, self.seen_stream_data, mode="eval")
        oie_predictions, oie_results, oie_results_all = self.evaluate(
            eval_dataloader=overall_instream_eval_dataloader, verbose=True)
        self.overall_eval_results["final_instream_test"] = oie_results
        # self.overall_eval_results["model0_instream_test"] = {"EM": float(
        #     np.mean([r["model0_instant_EM"] for r in self.online_eval_results]))}

        # Test the upstream forgetting eval.
        self.logger.info("Test the upstream forgetting eval.")
        # _, overall_upstream_eval_dataloader = self.get_dataloader(
        #     self.data_args, self.sampled_passes, mode="eval")
        oue_predictions, oue_results, oue_results_all = self.evaluate(
            eval_dataloader=self.forget_eval_loader, verbose=True)
        self.overall_eval_results["final_upstream_test"] = oue_results
        self.logger.info("Finish the final evaluation.")

    def _eval_forget_and_fix(self, examples, before_results, after_results):
        forgotten_examples = []
        retained_ids = []
        fixed_ids = []
        unfixed_ids = []
        for (_input, _truth, _id), em_before, em_after, before_pred, after_pred in zip(examples, before_results["metric_results_detailed"]["EM"], after_results["metric_results_detailed"]["EM"], before_results["predictions"], after_results["predictions"]):
            if em_before == 1:
                if em_after == 0:
                    forgotten_examples.append({"id": _id, "before_prediction": before_pred, "after_prediction": after_pred})
                else:
                    retained_ids.append(_id)

            if em_before == 0:
                if em_after == 1:
                    fixed_ids.append(_id)
                else:
                    unfixed_ids.append(_id)
        return forgotten_examples, retained_ids, fixed_ids, unfixed_ids
      # So the 0-th checkpoint should be the original base model.

    def _save_base_model(self, ckpt_name=None):
        output_dir = self.debugger_args.overtime_ckpt_dir
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

    def evaluate(self, eval_dataloader=None, verbose=False):
        """Evaluates the performance"""
        if not eval_dataloader:
            eval_dataloader = self.bug_eval_loaders[self.timecode]
        predictions = self.base_model_infer(eval_dataloader, verbose)
        assert len(predictions) == len(eval_dataloader)
        predictions = [p.strip() for p in predictions]
        results, results_all = evaluate_func(
            predictions, eval_dataloader.data, self.metric, return_all=True)

        return predictions, results, results_all

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
