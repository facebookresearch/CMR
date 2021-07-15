from semanticdebugger.debug_algs.continual_finetune_alg import ContinualFinetuning
from tqdm import tqdm
import random
import numpy as np
import torch
import transformers
from semanticdebugger.task_manager.eval_metrics import evaluate_func
import copy, pickle, os
from semanticdebugger.models.mybart import MyBart
from semanticdebugger.models import run_bart
from semanticdebugger.models.utils import (convert_model_to_single_gpu,
                                           freeze_embeds, trim_batch)
from argparse import Namespace

class KeyValueMemoryModule(object): 
    
    def __init__(self, logger, buffer=None):
        self.logger = logger        
        self.memory = {}
        self.keys_over_time = {}
        
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

    def store_examples(self, keys, examples, timecode=0):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        
        # update the memory dictionary
        for i, key in enumerate(keys):
            # numpy array cannot be used as key since it is non-hashable, hence convert it to bytes to use as key.
            values = list(examples[i])
            values.append(timecode)
            self.memory.update({key.tobytes(): tuple(values)})


    def query_examples(self, keys, k=32, timecode=0):
        """
        Returns samples from buffer using K-nearest neighbour approach
        """
        retrieved_examples = []
        # Iterate over all the input keys
        # to find neigbours for each of them
        
        # Only allow retrieving from the past memory. (due to offline evaluation)
        past_memory_keys = []
        for key, values in self.memory.items():
            if values[3]-1 <= timecode:
                past_memory_keys.append(key)
        past_memory_keys = np.frombuffer(np.asarray(past_memory_keys), dtype=np.float32).reshape(len(past_memory_keys), -1)       
        k = min(k, len(past_memory_keys))
        for key in keys:
            # compute similarity scores based on Euclidean distance metric
            similarity_scores = np.dot(past_memory_keys, key.T)
            K_neighbour_keys = past_memory_keys[np.argpartition(similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]
            # converts experiences into batch 
            # retrieved_examples.append(neighbours)
            retrieved_examples += neighbours

        return retrieved_examples

    def random_sample(self, sample_size):
        keys = random.sample(list(self.memory), sample_size)
        _inputs = [self.memory[k][0] for k in keys]
        _outputs = [self.memory[k][1] for k in keys]
        _ids = [self.memory[k][2] for k in keys]
        # _timecodes = [self.memory[k][3] for k in keys]
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
            "num_adapt_epochs"
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
                self.memroy_module.store_examples(key_vectors, bug_train_loader.data, timecode=self.timecode)
                self.logger.info("Finished.")
            

        self.memroy_module.save_memory_to_path(self.debugger_args.memory_path)
 

    def get_adapt_dataloaders(self, eval_dataloader=None, verbose=False):
        """Get the adapt_dataloader."""
        adapt_dataloaders = []
        num_batches = len(eval_dataloader.dataloader)
        example_batches = np.array_split(eval_dataloader.data, num_batches)
        for example_batch in tqdm(example_batches, desc="Retrieving Data from Memory", disable=not verbose):
            # self.logger.info("Memory Retrieving ...")
            # local adaptation for self.base_model of retrieved examples from memory.
            keys = self.memroy_module.encode_examples(example_batch)
            retrieved_examples = self.memroy_module.query_examples(keys, k=self.debugger_args.replay_size, timecode=self.timecode)
            replay_data_loader, _ = self.get_dataloader(self.data_args, retrieved_examples, mode="train")
            adapt_dataloaders.append(replay_data_loader) 
            # self.logger.info("Memory Retrieving Done ...")
        
        return adapt_dataloaders


    def base_model_infer(self, eval_dataloader, adapt_dataloaders, verbose=False):
        self.base_model.eval()
        model = self.base_model if self.n_gpu == 1 else self.base_model.module
        predictions = self.inference_with_adaptation(model, eval_dataloader, adapt_dataloaders, save_predictions=False, verbose=verbose, logger=self.logger, return_all=False, predictions_only=True, args=Namespace(quiet=True))
        return predictions


    def evaluate(self, eval_dataloader=None, verbose=False):
        """Evaluates the performance"""
        if not eval_dataloader:
            eval_dataloader = self.bug_eval_loaders[self.timecode]
        
        # prepare adapt_dataloaders
        adapt_dataloaders = self.get_adapt_dataloaders(eval_dataloader) 

        predictions = self.base_model_infer(eval_dataloader, adapt_dataloaders, verbose)
        assert len(predictions) == len(eval_dataloader)
        predictions = [p.strip() for p in predictions]
        results, return_all = evaluate_func(
            predictions, eval_dataloader.data, self.metric, return_all=True)

        return predictions, results, return_all

    def local_adaptation(self, model, adapt_dataloader):
        pad_token_id = self.tokenizer.pad_token_id
        base_weights = list(model.parameters())
        curr_weights = list(model.parameters())
        train_losses = []
        global_step = 0
        pad_token_id = self.tokenizer.pad_token_id
        super().debugger_setup(self.debugger_args)  # reset the optimizier and schduler
        model.train()
        model.zero_grad()
        for epoch_id in range(int(self.debugger_args.num_adapt_epochs)):
            for batch in tqdm(adapt_dataloader.dataloader, desc=f"Bug-fixing Epoch {epoch_id}", disable=True):
                if self.use_cuda:
                    # print(type(batch[0]), batch[0])
                    batch = [b.to(torch.device("cuda")) for b in batch]
                batch[0], batch[1] = trim_batch(
                    batch[0], pad_token_id, batch[1])
                batch[2], batch[3] = trim_batch(
                    batch[2], pad_token_id, batch[3])
                # this is the task loss w/o any regularization
                loss = model(input_ids=batch[0], attention_mask=batch[1],
                                       decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                                       is_training=True)
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.

                diff_loss = torch.Tensor([0]).to("cuda" if torch.cuda.is_available() else "cpu")
                # Iterate over base_weights and curr_weights and accumulate the euclidean norm
                # of their differences
                for base_param, curr_param in zip(base_weights, curr_weights):
                    diff_loss += (curr_param - base_param).pow(2).sum()
                loss = loss + 1e-3 * diff_loss

                train_losses.append(loss.detach().cpu())
                loss.backward()

                if global_step % self.debugger_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.debugger_args.max_grad_norm)
                    self.optimizer.step()    # We have accumulated enough gradients
                    self.scheduler.step()
                    model.zero_grad()
        return model

    def inference_with_adaptation(self, model, dev_data, adapt_dataloaders, save_predictions=False, verbose=False, args=None, logger=None, return_all=False, predictions_only=False):
        # model.eval()
        predictions = []
        bos_token_id = dev_data.tokenizer.bos_token_id
        loss = []   # if needed
        if args:
            quiet = args.quiet
        else:
            quiet = False
        if not quiet:
            logger.info("Starting inference ...")
        current_index = 0
        for batch in tqdm(dev_data.dataloader, desc="Infernece", disable=not verbose):
            ### Local Adaptation: Start ###
            _model = copy.deepcopy(model)
            adapt_dataloader = adapt_dataloaders[current_index]
            _model = self.local_adaptation(_model, adapt_dataloader)
            ### Local Adaptation: End ###

            _model.eval()

            ### Inference: Start ###
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            pad_token_id = dev_data.tokenizer.pad_token_id
            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            if return_all:
                pass
            outputs = _model.generate(input_ids=batch[0],
                                    attention_mask=batch[1],
                                    num_beams=dev_data.args.num_beams,
                                    max_length=dev_data.args.max_output_length,
                                    decoder_start_token_id=_model.config.bos_token_id,
                                    early_stopping=dev_data.gen_early_stop,)
            for input_, output in zip(batch[0], outputs):
                pred = dev_data.decode(output)
                predictions.append(pred)

            ### Inference: End ###
            current_index += 1


        if not quiet:
            logger.info("Starting inference ... Done")

        if predictions_only:
            return predictions
        if save_predictions:
            dev_data.save_predictions(predictions, )
        # logger.info("Starting evaluation metric ...")
        result = dev_data.evaluate(predictions, verbose=verbose)
        # logger.info("Starting evaluation metric ... Done!")
        if return_all:
            return predictions, result, loss
        return result