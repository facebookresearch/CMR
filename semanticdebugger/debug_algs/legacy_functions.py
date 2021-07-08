
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
