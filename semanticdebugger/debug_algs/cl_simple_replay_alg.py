from semanticdebugger.debug_algs.continual_finetune_alg import ContinualFinetuning
from tqdm import tqdm
import random 

class SimpleReplay(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "simple_replay"

    def _check_debugger_args(self):
        super()._check_debugger_args()
        required_atts = [ 
            "replay_size",
            ]
        assert all([hasattr(self.debugger_args, att) for att in required_atts])
 
    def random_replay(self):
        return random.sample(self.sampled_upstream_examples, self.debugger_args.replay_size)

    def online_debug(self):
        self.logger.info("Start Online Debugging")
        self.logger.info(f"Number of Batches of Bugs: {self.num_bug_batches}")
        self.logger.info(f"Bug Batch Size: {self.bug_batch_size}")
        self.logger.info(f"Replay Size: {self.debugger_args.replay_size}")
        self.timecode = 0

        if self.debugger_args.save_all_ckpts:
            # save the initial model as the 0-th model.
            self._save_base_model()

        for bug_train_loader in tqdm(self.bug_train_loaders, desc="Online Debugging", total=self.num_bug_batches):
            ############### Replay ############### 

            replayed_examples = self.random_replay()
            # reset the dataloader for merging the replayed examples
            bug_train_loader.data += replayed_examples
            bug_train_loader.load_dataset(self.tokenizer, skip_cache=True, quiet=True)
            bug_train_loader.load_dataloader()
            
            ############### CORE ###############
            # Fix the bugs by mini-batch based "training"
            self.logger.info("Start bug-fixing ....")
            self.fix_bugs(bug_train_loader)   # for debugging
            self.logger.info("Start bug-fixing .... Done!")
            ############### CORE ###############
            self.timecode += 1
            if self.debugger_args.save_all_ckpts:
                self._save_base_model()
                # Note that we save the model from the id=1.
                # So the 0-th checkpoint should be the original base model.