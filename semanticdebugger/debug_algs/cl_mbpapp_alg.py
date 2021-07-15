from semanticdebugger.debug_algs.continual_finetune_alg import ContinualFinetuning
from tqdm import tqdm
import random
import numpy as np
import torch
import transformers
from semanticdebugger.task_manager.eval_metrics import evaluate_func
import copy, pickle, os

class KeyValueMemoryModule(object): 
    
    def __init__(self, logger, buffer=None):
        self.logger = logger        
        self.memory = {}
        
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
        inputs = [d[0] for d in examples]
        with torch.no_grad():
            key_texts = self.get_key_content(inputs)    # only use the questions as the key text for encoding.
            inputs = self.tokenizer.batch_encode_plus(key_texts, return_tensors="pt", pad_to_max_length=True)
            last_hidden_states, _ = self.key_encoder(**inputs) 
            key_vectors = last_hidden_states[:, 0, :]
        return key_vectors.cpu().numpy()

    def store_examples(self, keys, examples):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        
        # update the memory dictionary
        for i, key in enumerate(keys):
            # numpy array cannot be used as key since it is non-hashable, hence convert it to bytes to use as key.
            self.memory.update({key.tobytes(): examples[i]})


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
        keys = random.sample(list(self.memory), sample_size)
        _inputs = [self.memory[k][0] for k in keys]
        _outputs = [self.memory[k][1] for k in keys]
        _ids = [self.memory[k][2] for k in keys]
        examples = list(zip(_inputs, _outputs, _ids))
        return examples

    def save_memory_to_path(self, memory_path):
        if self.memory is not None:
            with open(memory_path, "wb") as f:
                self.logger.info(f"Saving the memory to {memory_path}")
                pickle.dump(self.memory, f)

    def load_memory_from_path(self, memory_path):
        if os.path.exists(memory_path):
            with open(memory_path, "rb") as f:
                self.logger.info(f"Loading the memory from {memory_path}")
                self.memory = pickle.load(f)
                total_keys = len(self.memory.keys())
                # convert the keys from np.bytes to np.float32
                self.all_keys = np.frombuffer(
                    np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, -1)
        else:
            self.logger.info(f"Warning: {memory_path} doesn't exist.")

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
            "memory_path", # to save/load the memory module from disk
            ]
        assert all([hasattr(self.debugger_args, att) for att in required_atts])

    def debugger_setup(self, debugger_args):
        super().debugger_setup(debugger_args)
        
        # Initializing the Key-Value memory module for MBPA++
        self.memroy_module = KeyValueMemoryModule(self.logger)
        self.memroy_module.load_key_encoder(debugger_args.memory_key_encoder)
        self.memroy_module.load_memory_from_path(debugger_args.memory_path)
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
        last_steps = 0
        for bug_train_loader in tqdm(self.bug_train_loaders, desc="Online Debugging", total=self.num_bug_batches):

            if (self.model_update_steps - last_steps) >= self.debugger_args.replay_frequency:
                # sparse experience replay
                self.logger.info("Triggering Sampling from Memory and starting to replay.")
                retrieved_examples = self.memroy_module.random_sample(sample_size=self.debugger_args.replay_size)
                replay_data_loader, _ = self.get_dataloader(self.data_args, retrieved_examples, mode="train")
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
            if self.debugger_args.save_all_ckpts:
                self._save_base_model()
                # Note that we save the model from the id=1.
                # So the 0-th checkpoint should be the original base model.
            _max = 1000000
            flag_store_examples = bool(random.randrange(0, _max)/_max >= 1 - self.debugger_args.memory_store_rate)
            if flag_store_examples:
                self.logger.info("Saving examples to the memory.")
                key_vectors = self.memroy_module.encode_examples(bug_train_loader.data)
                self.memroy_module.store_examples(key_vectors, bug_train_loader.data)
                self.logger.info("Finished.")
            

        self.memroy_module.save_memory_to_path(self.debugger_args.memory_path)
 

    def evaluate(self, eval_dataloader=None, verbose=False):
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
