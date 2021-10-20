import random
import copy
from semanticdebugger.models.mybart import MyBart
from semanticdebugger.models import run_bart
import torch
import transformers
from semanticdebugger.models.utils import (convert_model_to_single_gpu,
                                           freeze_embeds, trim_batch)

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import more_itertools


def get_virtual_updated_model(cl_trainer, query_data_loader):
    before_model = copy.deepcopy(cl_trainer.base_model)
    virtual_adapt_args = copy.deepcopy(cl_trainer.data_args)
    virtual_adapt_args.train_batch_size = 4
    # change the batch size for the training.
    query_data_loader, _ = cl_trainer.get_dataloader(virtual_adapt_args, query_data_loader.data, mode="train") # fix of the order
    after_model = local_adaptation(cl_trainer, before_model, query_data_loader, diff_loss_weight=0)  
    del before_model
    return after_model

def get_top_interfered_examples(cl_trainer, K, candidate_examples, query_data_loader):
    """
    This is for the MIR method. 
    1) use query examples to train current_model for getting a virtual model.
    2) test the current_model and the virtual model seperately on the candidate examples
    3) compare the loss udpate of each example and rank them by the delta.
    4) return the top K examples with the largest positive loss changes.
    """
    # assert cl_trainer.name == "mir"

    cl_trainer.logger.info(
        f"get_top_interfered_examples: len(candidate_examples)={len(candidate_examples)};")


    if cl_trainer.debugger_args.mir_abalation_args == "random":
        cl_trainer.logger.info(f"ablation mode: randomly sample {K} examples from the candidate_examples")
        random.shuffle(candidate_examples)
        return candidate_examples[:K]

    ##################### Prepare the candidate examples as Memory Buffer #####################
    mlr_data_args = copy.deepcopy(cl_trainer.data_args)
    mlr_data_args.predict_batch_size = 8    # to get the loss for each example      # TODO: debug_MIR  
    # TODO: give the same random seed for selecting the same answer (if there are multiple answers)

    # only keep one possible correct answers for computing the loss consistnetly. 
    candidate_examples_single_ans = _keep_first_answer(candidate_examples)

    
    memory_buffer_loader, _ = cl_trainer.get_dataloader(
        mlr_data_args, candidate_examples_single_ans, mode="train", is_training=False) # fix of the order

    ##################### End #####################
    before_model = copy.deepcopy(cl_trainer.base_model)        
    before_losses = run_bart.inference(
        before_model, memory_buffer_loader, compute_loss=True, loss_only=True, logger=cl_trainer.logger)
    
    if cl_trainer.debugger_args.mir_abalation_args == "largest_beforeloss":
        after_losses = before_losses
    else:
        # virtual udpate 
        virtual_adapt_args = copy.deepcopy(cl_trainer.data_args)
        virtual_adapt_args.train_batch_size = 4
        # change the batch size for the training.
        query_data_loader, _ = cl_trainer.get_dataloader(virtual_adapt_args, query_data_loader.data, mode="train") # fix of the order
        
        after_model = local_adaptation(cl_trainer, before_model, query_data_loader, diff_loss_weight=0)  
        after_losses = run_bart.inference(
            after_model, memory_buffer_loader, compute_loss=True, loss_only=True, logger=cl_trainer.logger)
        # cl_trainer.logger.info(
        #     f"len(before_losses)={len(before_losses)}; len(after_losses)={len(after_losses)};")
        assert len(before_losses) == len(after_losses) == len(candidate_examples)
    # cl_trainer.logger.info(f"candidate_examples IDs: {[x[2] for x in candidate_examples]}")

    # it's a virtual update and we need to recover it.
    # del cl_trainer.base_model
    # del after_model
    # cl_trainer.base_model = before_model

    interference_scores = []
    for example, before_loss, after_loss in zip(candidate_examples, before_losses, after_losses):
        if cl_trainer.debugger_args.mir_abalation_args == "largest_afterloss":
            loss_delta = after_loss   # only for debugging MIR; biggest losers afterwards
        elif cl_trainer.debugger_args.mir_abalation_args == "largest_beforeloss":
            loss_delta = before_loss
        else:
            # standard MIR
            loss_delta = after_loss - before_loss
        interference_scores.append((example, loss_delta))

    # cl_trainer.logger.info(f"before_losses={before_losses}")

    # cl_trainer.logger.info(f"after_losses={after_losses}")

    # cl_trainer.logger.info(f"interference_scores={[x[1] for x in interference_scores]}")

    
    interference_scores.sort(key=lambda x: x[1], reverse=True)
        
    if cl_trainer.debugger_args.mir_abalation_args == "reverse":
        interference_scores.reverse() # only for debugging MIR. it's actually reverse=Yes
    
    top_K_examples = [x[0] for x in interference_scores][:K]

    # cl_trainer.logger.info(f"retrieved candidates ids = {[x[2] for x in top_K_examples]}")

    del before_model 
    del before_losses
    del after_model
    del after_losses
    del memory_buffer_loader
    return top_K_examples


def local_adaptation(cl_trainer, model, adapt_dataloader, diff_loss_weight=1e-3):
    pad_token_id = cl_trainer.tokenizer.pad_token_id
    base_weights = list(cl_trainer.base_model.parameters())
    curr_weights = list(model.parameters())
    global_step = 0
    pad_token_id = cl_trainer.tokenizer.pad_token_id

    # super().debugger_setup(cl_trainer.debugger_args)  # reset the optimizier and schduler

    model.train()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': cl_trainer.debugger_args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=cl_trainer.debugger_args.local_adapt_lr, eps=cl_trainer.debugger_args.adam_epsilon)

    # TODO: double check the decision about warm up for fine-tuning
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=cl_trainer.debugger_args.warmup_steps,
                                                num_training_steps=cl_trainer.debugger_args.total_steps)

    for epoch_id in range(int(cl_trainer.debugger_args.num_adapt_epochs)):
        for batch in tqdm(adapt_dataloader.dataloader, desc=f"Local Adaptation Epoch {epoch_id}", disable=False):
            global_step += 1
            if cl_trainer.use_cuda:
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
            if cl_trainer.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if diff_loss_weight != 0:
                diff_loss = torch.Tensor([0]).to("cuda" if torch.cuda.is_available() else "cpu")
                # Iterate over base_weights and curr_weights and accumulate the euclidean norm
                # of their differences
                for base_param, curr_param in zip(base_weights, curr_weights):
                    diff_loss += (curr_param - base_param).pow(2).sum()
                loss = loss + diff_loss_weight * diff_loss
            loss.backward()

            if global_step % cl_trainer.debugger_args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cl_trainer.debugger_args.max_grad_norm)
                optimizer.step()    # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()
    return model

def _keep_first_answer(examples_with_multiple_ans):
    examples_with_single_ans = []
    for item in examples_with_multiple_ans:
        examples_with_single_ans.append((item[0], item[1][0:1], item[2]))
    return examples_with_single_ans



class KeyValueMemoryModule(object):

    def __init__(self, logger):
        self.logger = logger
        self.memory = {}
        self.keys_over_time = {}
        self.memory_key_cache = {}
        self.memory_key_encoder = ""

    def load_key_encoder(self, memory_key_encoder='facebook/bart-base'):
        # https://huggingface.co/transformers/model_doc/bart.html#bartmodel
        # TODO: consider the SentenceBERT-like sentence encoders.
        self.memory_key_encoder = memory_key_encoder
        self.logger.info(
            f"Starting to load the key encoder ({memory_key_encoder}) for the memory module.")
        if "bart" in memory_key_encoder.lower():
            self.tokenizer = transformers.BartTokenizer.from_pretrained(memory_key_encoder)
            self.key_encoder = transformers.BartModel.from_pretrained(memory_key_encoder)
        elif "distilbert" in memory_key_encoder.lower():
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(memory_key_encoder)
            self.key_encoder = transformers.DistilBertModel.from_pretrained(memory_key_encoder)
        elif "roberta" in memory_key_encoder.lower():
            self.key_encoder = transformers.RobertaModel.from_pretrained(memory_key_encoder)
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(memory_key_encoder)
        elif "bert" in memory_key_encoder.lower():
            self.key_encoder = transformers.BertModel.from_pretrained(memory_key_encoder)
            self.tokenizer = transformers.BertTokenizer.from_pretrained(memory_key_encoder)

        self.key_encoder.cuda()
        self.logger.info(f"Finished.")
        return self.key_encoder, self.tokenizer

    def get_key_content(self, inputs):
        key_texts = []
        trigger_str = "Question: "
        for _input in inputs:
            start_ind = _input.index(trigger_str) + len(trigger_str)
            key_texts.append(_input[start_ind:])
        return key_texts

    def load_memory_key_cache(self, init_memory_cache_path):
        if os.path.exists(init_memory_cache_path):
            self.logger.info(f"Loading init_memory_cache_path from {init_memory_cache_path}")
            with open(init_memory_cache_path, "rb") as f:
                self.memory_key_cache = pickle.load(f)[self.memory_key_encoder]
        else:
            self.logger.info(f"Initializing an empty memory key cache.")
            self.memory_key_cache = None

    def encode_examples_for_caching(self, all_examples, batch_size=1, return_tensors=False):
        """
        Return key representation of the documents
        """
        # Freeze the weights of the key network to prevent key
        # representations from drifting as data distribution changes
        # with torch.no_grad():
        #     last_hidden_states, _
        # = self.key_encoder(contents, attention_mask=attn_masks)
        # Obtain key representation of every text content by selecting the its [CLS] hidden representation
        # keys = last_hidden_states[:, 0, :]
        all_vectors = {}
        all_tensors = []
        batches = list(more_itertools.chunked(all_examples, batch_size))
        for examples in tqdm(batches, desc="Caching the examples"):
            inputs = [d[0] for d in examples]
            with torch.no_grad():
                # only use the questions as the key text for encoding.
                key_texts = self.get_key_content(inputs)
                inputs = self.tokenizer.batch_encode_plus(
                    key_texts, return_tensors="pt", pad_to_max_length=True)
                input_ids = inputs["input_ids"].to(torch.device("cuda"))
                attention_mask = inputs["attention_mask"].to(torch.device("cuda"))
                # last_hidden_states, _ = self.key_encoder(**inputs)
                results = self.key_encoder(input_ids, attention_mask)
                last_hidden_states = results[0]
                key_vectors = last_hidden_states[:, 0, :]
                key_vectors_npy = key_vectors.cpu().numpy()
                all_tensors += list(key_vectors)
            for key_text, key_vector in zip(key_texts, key_vectors_npy):
                all_vectors[key_text] = key_vector
        if return_tensors:
            return all_tensors
        return all_vectors

    def encode_examples(self, examples, use_random_keys=False):
        """
        Return key representation of the documents
        """

        inputs = [d[0] for d in examples]
        # only use the questions as the key text for encoding.
        key_texts = self.get_key_content(inputs)
        key_vectors = None

        if use_random_keys:
            self.logger.info("Using randomly generated memory keys for ER and MIR.")
            key_vectors = np.random.rand(len(examples), 128)
            return key_vectors

        if self.memory_key_cache:
            # self.logger.info("Using the cache.")
            key_vectors = []
            for key_text in key_texts:
                assert key_text in self.memory_key_cache, key_text
                key_vectors.append(self.memory_key_cache[key_text])
        else:
            # on the fly
            with torch.no_grad():
                inputs = self.tokenizer.batch_encode_plus(
                    key_texts, return_tensors="pt", pad_to_max_length=True)
                input_ids = inputs["input_ids"].to(torch.device("cuda"))
                attention_mask = inputs["attention_mask"].to(torch.device("cuda"))
                # last_hidden_states, _ = self.key_encoder(**inputs)
                results = self.key_encoder(input_ids, attention_mask)
                last_hidden_states = results[0]
                key_vectors = last_hidden_states[:, 0, :]
                key_vectors = key_vectors.cpu().numpy()
        return key_vectors

    def store_examples(self, keys, examples, timecode=0):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        assert len(keys) == len(examples)
        # update the memory dictionary
        for i, key in enumerate(keys):
            # numpy array cannot be used as key since it is non-hashable, hence convert it to bytes to use as key.
            values = list(examples[i])
            values.append(timecode)
            self.memory.update({key.tobytes(): tuple(values)})

    def query_examples(self, keys, past_memory_keys, k=32):
        """
        Returns samples from buffer using K-nearest neighbour approach
        """
        retrieved_examples = []
        # Iterate over all the input keys
        # to find neigbours for each of them

        k = min(k, len(past_memory_keys))
        for key in keys:
            # compute similarity scores based on Euclidean distance metric
            similarity_scores = np.dot(past_memory_keys, key.T)
            K_neighbour_keys = past_memory_keys[np.argpartition(similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]
            # converts experiences into batch
            # retrieved_examples.append(neighbours)
            retrieved_examples += neighbours
        # self.logger.info(f"Retrieved {len(retrieved_examples)} examples from memory; {len(retrieved_examples)/len(keys)} examples per key.")
        return retrieved_examples

    def random_sample(self, sample_size):
        sample_size = min(len(self.memory), sample_size)
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




def KVMemory_init():
    from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
    import argparse
    import json
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--memory_key_encoder', type=str, default="facebook/bart-base")
    parser.add_argument('--init_memory_cache_path', type=str,
                        default="bug_data/memory_key_cache.pkl")
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument("--bug_stream_json_path",
                        default="bug_data/mrqa_naturalquestions_dev.static_bug_stream.json")
    parser.add_argument("--pass_pool_jsonl_path",
                        default="bug_data/mrqa_naturalquestions_dev.sampled_pass.jsonl")
    parser.add_argument("--sampled_upstream_json_path",
                        default="bug_data/mrqa_naturalquestions.sampled_upstream.jsonl")

    args = parser.parse_args()

    log_filename = f'logs/memory_cache_building_{args.memory_key_encoder.replace("/", "_")}.log'

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_filename),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    cl_trainer = ContinualFinetuning(logger)

    # Load bugs
    with open(args.bug_stream_json_path) as f:
        bug_stream = json.load(f)
        all_examples = []
        for bug_batch in tqdm(bug_stream, desc="Creating the bug data loaders."):
            formatted_bug_batch = cl_trainer.data_formatter(bug_batch)
            all_examples += formatted_bug_batch

    # Load pass cases
    with open(args.pass_pool_jsonl_path) as f:
        pass_examples = [json.loads(line) for line in set(f.read().splitlines())]
        all_examples += cl_trainer.data_formatter(pass_examples)
    memory_module = KeyValueMemoryModule(logger)

    logger.info(f"All examples: {len(all_examples)}")
    memory_module.load_key_encoder(memory_key_encoder=args.memory_key_encoder)
    all_key_vectors = memory_module.encode_examples_for_caching(
        all_examples, batch_size=args.batch_size)

    logger.info(
        f"all_key_vectors.shape: {len(all_key_vectors)} x {len(all_key_vectors[list(all_key_vectors.keys())[0]])}")

    if os.path.exists(args.init_memory_cache_path):
        with open(args.init_memory_cache_path, "rb") as f:
            memory_key_cache = pickle.load(f)
    else:
        memory_key_cache = {}
    memory_key_cache[args.memory_key_encoder] = all_key_vectors

    with open(args.init_memory_cache_path, "wb") as f:
        pickle.dump(memory_key_cache, f)
        logger.info(f"Saved the cache to {f.name}")
