from semanticdebugger.debug_algs.continual_finetune_alg import ContinualFinetuning

class OfflineDebugger(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "offline_debug"

    def _check_debugger_args(self):
        super()._check_debugger_args()
        required_atts = [
            # additional hyper parameters
            "use_sampled_upstream",]
        assert all([hasattr(self.debugger_args, att) for att in required_atts])

    def offline_debug(self):
        """"This function is to generate the bound when fixing the errors offline."""
        self.logger.info("Start Offline Debugging")
        self.timecode = -1
        if self.debugger_args.use_sampled_upstream:
            merged_examples = self.all_bug_examples + self.sampled_upstream_examples
            dl, _ = self.get_dataloader(self.data_args, merged_examples, mode="train")
        else:
            assert self.bug_all_train_loader is not None 
            dl = self.bug_all_train_loader
        self.fix_bugs(dl, quiet=False)
        if self.debugger_args.save_all_ckpts:
            self._save_base_model(ckpt_name="offline_debug")
        
