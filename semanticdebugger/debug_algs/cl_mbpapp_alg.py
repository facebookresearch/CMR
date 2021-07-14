from semanticdebugger.debug_algs.continual_finetune_alg import ContinualFinetuning
from tqdm import tqdm
import random
import numpy as np
import torch
import transformers
from semanticdebugger.task_manager.eval_metrics import evaluate_func
from semanticdebugger.models import run_bart
from argparse import Namespace

class KeyValueMemoryModule(object): 
    
    def __init__(self, logger, buffer=None):
        self.logger = logger
        if buffer is None:
            self.memory = {}
        else:
            self.memory = buffer
            total_keys = len(buffer.keys())
            # convert the keys from np.bytes to np.float32
            self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, 768)

    def load_key_encoder(self, memory_key_encoder='facebook/bart-base'):
        # https://huggingface.co/transformers/model_doc/bart.html#bartmodel
        # TODO: consider the SentenceBERT-like sentence encoders.

        self.logger.info(f"Starting to load the key encoder ({memory_key_encoder}) for the memory module.")
        if "bart" in memory_key_encoder.lower():
            self.tokenizer = transformers.BartTokenizer.from_pretrained(memory_key_encoder)
            self.key_encoder = transformers.BartModel.from_pretrained(memory_key_encoder)
        elif "bert" in memory_key_encoder.lower():
            self.key_encoder = transformers.BertModel.from_pretrained(memory_key_encoder)
        self.logger.info(f"Finished.")

    def get_key_content(self, inputs):
        key_texts = []
        trigger_str = "Question: "
        for _input in inputs:
            start_ind = _input.index(trigger_str) + len(trigger_str)
            key_texts.append(_input[start_ind:])
        return key_texts

    def encode_examples(self, examples):
        """
        Return key representation of the documents
        """
        # Freeze the weights of the key network to prevent key
        # representations from drifting as data distribution changes
        # with torch.no_grad():
        #     last_hidden_states, _ = self.key_encoder(contents, attention_mask=attn_masks)
        # Obtain key representation of every text content by selecting the its [CLS] hidden representation
        # keys = last_hidden_states[:, 0, :]
        inputs, outputs = examples
        with torch.no_grad():
            key_texts = self.get_key_content(inputs)    # only use the questions as the key text for encoding.
            inputs = self.tokenizer.batch_encode_plus(key_texts, return_tensors="pt")
            last_hidden_states, _ = self.key_encoder(**inputs) 
            key_vectors = last_hidden_states[:, 0, :]
        return key_vectors

    def store_examples(self, keys, examples):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        inputs, outputs = examples
        # update the memory dictionary
        for i, key in enumerate(keys):
            # numpy array cannot be used as key since it is non-hashable, hence convert it to bytes to use as key.
            self.memory.update({key.tobytes(): {"input":inputs[i], "output": outputs[i]}})


    def query_examples(self, keys, k=32):
        """
        Returns samples from buffer using K-nearest neighbour approach
        """
        retrieved_examples = []
        # Iterate over all the input keys
        # to find neigbours for each of them
        for key in keys:
            # compute similarity scores based on Euclidean distance metric
            similarity_scores = np.dot(self.all_keys, key.T)
            K_neighbour_keys = self.all_keys[np.argpartition(similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]
            # converts experiences into batch 
            retrieved_examples.append(neighbours)

        return retrieved_examples

    def random_sample(self, sample_size):
        keys = random.sample(list(self.memory),sample_size)
        contents = np.array([self.memory[k][0] for k in keys])
        attn_masks = np.array([self.memory[k][1] for k in keys])
        labels = np.array([self.memory[k][2] for k in keys])
        return (torch.LongTensor(contents), torch.LongTensor(attn_masks), torch.LongTensor(labels)) 


class MBPAPlusPlus(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "mbpa++"

    def _check_debugger_args(self):
        super()._check_debugger_args()
        required_atts = [
            "replay_size",
            "replay_frequency", # 
            "memory_key_encoder", # 'bert-base-uncased' by default
            "memory_store_rate", # 0, 0.1, 1 etc.
            ]
        assert all([hasattr(self.debugger_args, att) for att in required_atts])

    def debugger_setup(self, debugger_args):
        super().debugger_setup(debugger_args)
        
        # Initializing the Key-Value memory module for MBPA++
        self.memroy = KeyValueMemoryModule(self.logger)
        self.memroy.load_key_encoder(debugger_args.memory_key_encoder)
        return

    def online_debug(self):
        self.logger.info("Start Online Debugging")
        self.logger.info(f"Number of Batches of Bugs: {self.num_bug_batches}")
        self.logger.info(f"Bug Batch Size: {self.bug_batch_size}")
        self.logger.info(f"Replay Size: {self.debugger_args.replay_size}")
        self.logger.info(f"Replay Frequency: {self.debugger_args.replay_frequency}")
        self.timecode = 0

        if self.debugger_args.save_all_ckpts:
            # save the initial model as the 0-th model.
            self._save_base_model()
        
        # For the initial memory.
        # TODO: sample and save to the memory.

        for bug_train_loader in tqdm(self.bug_train_loaders, desc="Online Debugging", total=self.num_bug_batches):

            if (self.model_update_steps + 1) % self.debugger_args.replay_frequency == 0:
                # start local adaptation
                self.logger.info("Triggering Sampling from Memory and starting to replay.")
                retrieved_examples = self.memroy.random_sample(sample_size=self.debugger_args.replay_size)
                replay_data_loader, _ = self.get_dataloader(self.bug_data_args, retrieved_examples, mode="train")
                self.fix_bugs(replay_data_loader, quiet=False)
                self.logger.info("Replay-Training done.")
                # TODO: get the dataloader and then call the fix the bugs.
            

            ############### CORE START ###############
            # Fix the bugs by mini-batch based "training"
            self.logger.info("Start bug-fixing ....")
            self.fix_bugs(bug_train_loader)   # for debugging
            self.logger.info("Start bug-fixing .... Done!")
            ############### CORE END ###############
            self.timecode += 1
            if self.debugger_args.save_all_ckpts:
                self._save_base_model()
                # Note that we save the model from the id=1.
                # So the 0-th checkpoint should be the original base model.
            _max = 1000000
            flag_store_examples = bool(random.randrange(0, _max)/_max >= 1 - self.debugger_args.memory_store_rate)
            if flag_store_examples:
                self.logger.info("Saving examples to the memory.")
                self.memroy.store_examples(bug_train_loader.data)
    
    # def base_model_infer(self, eval_dataloader, verbose=False):
    #     self.base_model.eval()
    #     model = self.base_model if self.n_gpu == 1 else self.base_model.module
    #     predictions = run_bart.inference(model, eval_dataloader, save_predictions=False, verbose=verbose,
    #                                      logger=self.logger, return_all=False, predictions_only=True, args=Namespace(quiet=True))
    #     return predictions

    def evaluate(self, eval_dataloader=None, verbose=False):
        """Evaluates the performance"""
        if not eval_dataloader:
            eval_dataloader = self.bug_eval_loaders[self.timecode]
        
        # TODO: local adaptation of retrieved examples from memory.

        predictions = self.base_model_infer(eval_dataloader, verbose)
        assert len(predictions) == len(eval_dataloader)
        predictions = [p.strip() for p in predictions]
        results, return_all = evaluate_func(
            predictions, eval_dataloader.data, self.metric, return_all=True)