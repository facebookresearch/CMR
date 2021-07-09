from semanticdebugger.debug_algs.continual_finetune_alg import ContinualFinetuning

class OfflineDebugger(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "offline_debug"

    def offline_debug(self):
        """"This function is to generate the bound when fixing the errors offline."""
        assert self.bug_all_train_loader is not None 
        self.logger.info("Start Offline Debugging")
        self.timecode = -1
        self.fix_bugs(self.bug_all_train_loader, quiet=False)
        if self.debugger_args.save_all_ckpts:
            self._save_base_model(ckpt_name="offline_debug")
    
    