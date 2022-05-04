# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# TODO: remove this as we have the offline evaluation function now.
def _eval_before_fixing(self):
    # Before Bug-Fixing
    assert self.online_debug_results is not None
    bug_eval_loader = self.bug_eval_loaders[self.timecode]
    bug_before_predictions, bug_before_results, bug_before_results_all = self.evaluate(
        bug_eval_loader)
    self.logger.info("-"*10+f"Timecode: {self.timecode}"+"-"*10)
    self.logger.info(
        f"Before Bug-fixing the results on bug-batch-{self.timecode} = {bug_before_results}")
    if len(self.online_debug_results["res_on_passes"]) == 0:
        pass_before_predictions, pass_before_results, pass_before_results_all = self.evaluate(
            self.forget_eval_loader)
        self.online_debug_results["res_on_passes"].append(
            (pass_before_results, pass_before_results_all))
    else:
        pass_before_predictions = None  # TODO:
        pass_before_results, pass_before_results_all = self.online_debug_results[
            "res_on_passes"][-1]
    self.logger.info(
        f"Before Bug-fixing the results on the sampled pass cases = {pass_before_results}")
    return bug_before_results, bug_before_results_all, pass_before_results, pass_before_results_all

# TODO: remove this as we have the offline evaluation function now.
def _eval_after_fixing(self, bug_before_results, bug_before_results_all, pass_before_results, pass_before_results_all):
    # After Bug-Fixing
    assert self.online_debug_results is not None
    bug_eval_loader = self.bug_eval_loaders[self.timecode]
    bug_after_predictions, bug_after_results, bug_after_results_all = self.evaluate(
        bug_eval_loader)
    self.logger.info(
        f"After Bug-fixing the results on bug-batch-{self.timecode} = {bug_after_results}")
    pass_after_predictions, pass_after_results, pass_after_results_all = self.evaluate(
        self.forget_eval_loader)
    self.logger.info(
        f"After Bug-fixing the results on the sampled pass cases = {pass_after_results}")

    # Log the overall results
    self.online_debug_results["res_on_bugs"].append(
        (bug_before_results, bug_after_results))
    self.online_debug_results["res_on_passes"].append(
        (pass_after_results, pass_after_results_all))
    self._check_fixing(
        bug_eval_loader, bug_before_results_all, bug_after_results_all)
    self._check_forgetting(pass_before_results_all, pass_after_results_all)

    if self.debugger_args.overtime_overall_bug_eval:
        all_bug_after_predictions, all_bug_after_results, all_bug_after_results_all = self.evaluate(
            self.bug_all_eval_loader)
        self.logger.info(
            f"Current Overall Bug-fixing Results = {all_bug_after_results}")
        self.online_debug_results["overtime_all_bug_eval"].append(
            all_bug_after_results)

# TODO: remove this as we have the offline evaluation function now.
def _eval_overall_bugs(self):
    all_bug_after_predictions, all_bug_after_results, all_bug_after_results_all = self.evaluate(
        self.bug_all_eval_loader)
    self.online_debug_results["final_all_bug_eval"] = all_bug_after_results
    self.logger.info(
        f"Final Overall Bug-fixing Results = {all_bug_after_results}")



# TODO: move to evaluation analysis part.
def _check_fixing(self, bug_eval_loader, bug_before_results_all, bug_after_results_all):
    # Log the specific fixed bugs and forget examples
    em_prefixed_bugs = []
    f1_prefixed_bugs = []
    em_fixed_bugs = []
    f1_fixed_bugs = []
    assert len(bug_eval_loader.data) == len(
        bug_before_results_all["EM"]) == len(bug_after_results_all["EM"])
    for ind in range(len(bug_eval_loader.data)):
        em_before = bug_before_results_all["EM"][ind]
        em_after = bug_after_results_all["EM"][ind]
        f1_before = bug_before_results_all["QA-F1"][ind]
        f1_after = bug_after_results_all["QA-F1"][ind]
        uuid = bug_eval_loader.data[ind][2]  # (input, output, uuid)
        if em_before == 1:
            em_prefixed_bugs.append(uuid)
        if f1_after > 0.5:
            f1_prefixed_bugs.append(uuid)
        if em_before == 0 and em_after == 1:
            em_fixed_bugs.append(uuid)
        if f1_before < 0.5 and f1_after > 0.5 and f1_after-f1_before >= 0.25:
            f1_fixed_bugs.append(uuid)

    self.online_debug_results["em_fixed_bugs"].append(em_fixed_bugs)
    self.online_debug_results["f1_fixed_bugs"].append(f1_fixed_bugs)
    self.online_debug_results["em_prefixed_bugs"].append(em_prefixed_bugs)
    self.online_debug_results["f1_prefixed_bugs"].append(f1_prefixed_bugs)
    self.logger.info(
        f"Number of em_prefixed_bugs = {len(em_prefixed_bugs)}; Number of f1_prefixed_bugs = {len(f1_prefixed_bugs)}")
    self.logger.info(
        f"Number of em_fixed_bugs = {len(em_fixed_bugs)}; Number of f1_fixed_bugs = {len(f1_fixed_bugs)}")

# TODO: move to evaluation analysis part.
def _check_forgetting(self, pass_before_results_all, pass_after_results_all):
    # log the forgotten bugs
    em_forgotten_passes = []

    for ind in range(len(self.forget_eval_loader.data)):
        em_before = pass_before_results_all["EM"][ind]
        em_after = pass_after_results_all["EM"][ind]
        # f1_before = pass_before_results_all["QA-F1"][ind]
        # f1_after = pass_after_results_all["QA-F1"][ind]
        uuid = self.forget_eval_loader.data[ind][2]  # (input, output, uuid)
        if em_before == 1 and em_after == 0:
            em_forgotten_passes.append(uuid)

    self.online_debug_results["forgotten_passes"].append(
        em_forgotten_passes)
    self.logger.info(
        f"Number of em_forgotten_passes = {len(em_forgotten_passes)}.")
    # self.logger.info(f"UUIDS of fixed bugs = {em_fixed_bugs}")


def evaluate_v1(self, eval_dataloader=None, verbose=False):
    """Evaluates the performance"""

    # backup the base model.
    self.logger.info("Backing up the base model ...")
    base_model_backup = copy.deepcopy(self.base_model)
    self.logger.info("Backking up the base model ... Done!")
    
    
    self.logger.info("Memory Retrieving ...")
    # local adaptation for self.base_model of retrieved examples from memory.
    keys = self.memroy_module.encode_examples(eval_dataloader.data)
    retrieved_examples = self.memroy_module.query_examples(keys, k=self.debugger_args.replay_size)
    replay_data_loader, _ = self.get_dataloader(self.data_args, retrieved_examples, mode="train")
    self.logger.info("Memory Retrieving Done ...")
    
    self.logger.info("Temp local adaptation ...")
    self.fix_bugs(replay_data_loader)  # local adaptation
    self.logger.info("Temp local adaptation ... Done")


    # get inference as usual.

    predictions, results, return_all = super().evaluate(eval_dataloader=None, verbose=False)

    del self.base_model
    
    self.base_model = base_model_backup # restore to the original base_model

    return predictions, results, return_all



### Check the accumulative results. ###
if (self.data_args.accumulate_eval_freq > 0 and (self.timecode + 1) % self.data_args.accumulate_eval_freq == 0):
    accumu_EM, forgotten_ids, fixed_ids, total_len = self.get_accumulative_results()
    result_dict["accumulative_EM"] = accumu_EM
    result_dict["accumulative_forgotten_ids"] = forgotten_ids
    result_dict["accumulative_fixed_ids"] = fixed_ids
    result_dict["accumulative_forgotten_rate"] = len(forgotten_ids) / total_len
    result_dict["accumulative_fixed_rate"] = len(fixed_ids) / total_len

    self.logger.info(" ")
    self.logger.info(
        f"Doing-Nothing Accumulative EM: {self.accumulate_doing_nothing_EM[self.timecode]}")
    self.logger.info(f"My Accumulative EM: {accumu_EM}")
    self.logger.info(
        f"accumulative_forgotten_rate: {result_dict['accumulative_forgotten_rate']}")
    self.logger.info(
        f"accumulative_fixed_rate: {result_dict['accumulative_fixed_rate']}")


def get_accumulative_results(self):
    EMs = []
    forgotten_ids = []
    fixed_ids = []
    total_len = 0
    for data_eval_loader in tqdm(self.data_eval_loaders[:self.timecode], desc="Evaluate Accumulative Results"):
        predictions, results, results_all = self.evaluate(data_eval_loader)
        EMs.append(results["EM"])
        for (_, _, _id), em in zip(data_eval_loader.data, results_all["EM"]):
            if _id in self.all_initial_error_ids and em == 1:
                fixed_ids.append(_id)
            if _id in self.all_initial_pass_ids and em == 0:
                forgotten_ids.append(_id)
            total_len += 1
    return float(np.mean(EMs)), forgotten_ids, fixed_ids, total_len


def single_timecode_eval(self, timecode):
    """Used only for offline eval of a single checkpoint of a specific timecode."""
    self.timecode = timecode
    result_dict = {}   # initialize for the given time code

    self.logger.info("Start the Overall Error-Fixing Results....")
    # Overall Error-Fixing Results
    eval_results_overall_bug = self.evaluate(
        self.bug_all_eval_loader, verbose=True)
    result_dict["eval_results_overall_bug"] = _pack_as_dict(
        *eval_results_overall_bug)
    self.logger.info("Start the Overall Error-Fixing Results....Done")

    self.logger.info(
        "Start the Overall Forgetting Results (Knowledge Retain Acc)....")
    # Overall Forgetting Results (Knowledge Retain Acc)
    eval_results_overall_forget = self.evaluate(
        self.forget_eval_loader, verbose=True)
    result_dict["eval_results_overall_forget"] = _pack_as_dict(
        *eval_results_overall_forget)
    self.logger.info(
        "Start the Overall Forgetting Results (Knowledge Retain Acc)....Done")

    if self.name == "offline_debug":
        # only overall evaluation for the offline debugging.
        return result_dict

    # Error-Fixing performance on the current batch of errors.
    if self.timecode > 0:
        self.logger.info(
            "Start Error-Fixing performance on the Current batch of errors.....")
        bug_eval_loader = self.bug_eval_loaders[self.timecode-1]
        eval_results_current_errors = self.evaluate(bug_eval_loader)
        result_dict["eval_results_current_errors"] = _pack_as_dict(
            *eval_results_current_errors)
        self.logger.info(
            "Start Error-Fixing performance on the Current batch of errors.....Done")

    # Error-Fixing performance on the next batch of errors. (for the computation of real responsive efr)
    if self.timecode < len(self.bug_eval_loaders):
        self.logger.info(
            "Start Error-Fixing performance on the Next batch of errors.....")
        bug_eval_loader = self.bug_eval_loaders[self.timecode]
        eval_results_next_errors = self.evaluate(bug_eval_loader)
        result_dict["eval_results_next_errors"] = _pack_as_dict(
            *eval_results_next_errors)
        self.logger.info(
            "Start Error-Fixing performance on the Next batch of errors.....Done")

    return result_dict



def load_data_static(self, data_args):
    self.data_args = data_args
    self._check_data_args()
    # Load bug stream
    with open(data_args.bug_stream_json_path) as f:
        bug_stream = json.load(f)
    self.bug_stream = bug_stream
    self.num_bug_batches = len(bug_stream)
    self.bug_batch_size = len(bug_stream[0])
    # Create data loaders for each error batch.
    all_formatted_bugs = []
    for bug_batch in tqdm(self.bug_stream, desc="Creating the bug data loaders."):
        formatted_bug_batch = self.data_formatter(bug_batch)
        all_formatted_bugs += formatted_bug_batch
        train_bug_dataloader, eval_bug_dataloader = self.get_dataloader(
            data_args, formatted_bug_batch, mode="both")
        self.bug_train_loaders.append(train_bug_dataloader)
        self.bug_eval_loaders.append(eval_bug_dataloader)
    assert len(self.bug_train_loaders) == self.num_bug_batches
    self.all_bug_examples = all_formatted_bugs
    # Create the all bug loaders.
    self.bug_all_train_loader, self.bug_all_eval_loader = self.get_dataloader(
        data_args, all_formatted_bugs, mode="both")

    # Create the pass pool evaluation loader for the final forgetting issue.

    if data_args.upstream_eval_data:
        # Create loaders for the sampled pass examples
        with open(data_args.upstream_eval_data) as f:
            pass_examples = [json.loads(line)
                                for line in set(f.read().splitlines())]
        self.sampled_passes = pass_examples
        pass_examples = self.data_formatter(pass_examples)
        _, self.forget_eval_loader = self.get_dataloader(
            data_args, pass_examples, mode="eval")

    if data_args.sampled_upstream_json_path:
        # Create loaders for the sampled pass examples
        with open(data_args.sampled_upstream_json_path) as f:
            sampled_upstream_examples = [json.loads(line)
                                            for line in set(f.read().splitlines())]
        self.sampled_upstream_examples = self.upstream_data_formatter(
            sampled_upstream_examples)
        # self.sampled_upstream_trainloader, self.sampled_upstream_evalloader = self.get_dataloader(
        #     data_args, sampled_upstream_examples, mode="eval")
        
    return


def online_debug_static(self):
    """For the static error stream."""
    self.logger.info("Start Online Debugging with Static Error Mode")
    self.logger.info(f"Number of Batches of Bugs: {self.num_bug_batches}")
    self.logger.info(f"Bug Batch Size: {self.bug_batch_size}")
    self.timecode = 0

    if self.debugger_args.save_ckpt_freq:
        # save the initial model as the 0-th model.
        self._save_base_model()

    for bug_train_loader in tqdm(self.bug_train_loaders, desc="Online Debugging (Static)", total=self.num_bug_batches):
        ############### CORE ###############
        # Fix the bugs by mini-batch based "training"
        self.logger.info(f"Start bug-fixing .... Timecode: {self.timecode}")
        self.fix_bugs(bug_train_loader)   # for debugging
        self.logger.info("Start bug-fixing .... Done!")
        ############### CORE ###############
        self.timecode += 1
        if self.debugger_args.save_ckpt_freq:
            self._save_base_model()
            # Note that we save the model from the id=1.
        
    
# cmr/debug_algs/cl_mbcl_alg.py
def online_debug_static(self):
    self.logger.info("Start Online Debugging")
    self.logger.info(f"Number of Batches of Bugs: {self.num_bug_batches}")
    self.logger.info(f"Bug Batch Size: {self.bug_batch_size}")
    self.logger.info(f"Replay Size: {self.debugger_args.replay_size}")
    self.logger.info(f"Replay Frequency: {self.debugger_args.replay_frequency}")
    self.timecode = 0

    if self.debugger_args.save_ckpt_freq:
        # save the initial model as the 0-th model.
        self._save_base_model()

    # For the initial memory.
    # TODO: sample and save to the memory.
    last_steps = 0
    for bug_train_loader in tqdm(self.bug_train_loaders, desc="Online Debugging", total=self.num_bug_batches):

        if (self.model_update_steps - last_steps) >= self.debugger_args.replay_frequency \
                and self.debugger_args.replay_frequency > 0 and self.debugger_args.replay_size > 0:
            # sparse experience replay
            self.logger.info("Triggering Sampling from Memory and starting to replay.")
            retrieved_examples = self.memroy_module.random_sample(
                sample_size=self.debugger_args.replay_size)
            replay_data_loader, _ = self.get_dataloader(
                self.data_args, retrieved_examples, mode="train")
            self.fix_bugs(replay_data_loader)  # sparse replay
            self.logger.info("Replay-Training done.")

        last_steps = self.model_update_steps
        ############### CORE START ###############
        # Fix the bugs by mini-batch based "training"
        self.logger.info(f"Start bug-fixing .... Timecode: {self.timecode}")
        self.fix_bugs(bug_train_loader)   # for debugging
        self.logger.info("Start bug-fixing .... Done!")
        ############### CORE END ###############
        self.timecode += 1
        if self.debugger_args.save_ckpt_freq:
            self._save_base_model()
            # Note that we save the model from the id=1.
            # So the 0-th checkpoint should be the original base model.
        _max = 1000000
        flag_store_examples = bool(random.randrange(0, _max)/_max >=
                                    1 - self.debugger_args.memory_store_rate)
        if flag_store_examples:
            self.logger.info("Saving examples to the memory.")
            key_vectors = self.memroy_module.encode_examples(bug_train_loader.data, use_random_keys=bool(self.name in ["er", "mir"]))
            self.memroy_module.store_examples(
                key_vectors, bug_train_loader.data, timecode=self.timecode)
            self.logger.info("Finished.")

    self.memroy_module.save_memory_to_path(self.debugger_args.memory_path)

