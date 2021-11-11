from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
from tqdm import tqdm 
import json 
import random 

class OfflineDebugger(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "offline_debug"

    def _check_debugger_args(self):
        super()._check_debugger_args()
        required_atts = [
            # additional hyper parameters
            "offline_retrain_upstream",]
        assert all([hasattr(self.debugger_args, att) for att in required_atts])

    def _get_all_init_errors(self):
        data_args = self.data_args
        all_init_errors = []
        for data_batch in tqdm(self.data_stream, desc="Creating the data loaders."):
            if data_args.max_timecode > 0 and len(self.data_eval_loaders) >= data_args.max_timecode:
                break
            all_init_errors += [item for item in data_batch if item["init_status"] == "error"]
        all_init_errors = self.data_formatter(all_init_errors) 
        return all_init_errors

    def offline_debug(self):
        """"This function is to generate the bound when fixing the errors offline."""
        
        self.logger.info("Start Offline Debugging")
        self.timecode = -1
        
        # TODO: get the all_bug_examples 
        init_errors = self._get_all_init_errors()
        # get the upstream examples
        with open(self.data_args.upstream_data_path) as f:
            upstream_memory_examples = [json.loads(line)for line in set(f.read().splitlines())]            
            upstream_memory_examples = self.upstream_data_formatter(upstream_memory_examples)

        if self.debugger_args.offline_retrain_upstream: 
            merged_examples = init_errors + upstream_memory_examples
        else:
            merged_examples = init_errors
        
        # dl, _ = self.get_dataloader(self.data_args, merged_examples, mode="train")
        # self.fix_bugs(dl, quiet=False)
        # self._save_base_model(ckpt_name="offline")
        
