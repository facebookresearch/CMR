from argparse import Namespace
from semanticdebugger.debug_algs.cl_utils import _keep_first_answer

from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
from tqdm import tqdm
import torch
from semanticdebugger.debug_algs.index_based.index_utils import get_bart_dual_representation
from semanticdebugger.models.utils import trim_batch
import json


from transformers.modeling_bart import _prepare_bart_decoder_inputs
import numpy as np
import faiss
import pickle
import random


class BaseMemoryManager():
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.name = "base_memory_manager"
        self.memory_examples = {}

    def get_memory_size(self):
        return len(self.memory_examples)

    def _load_init_memory_examples(self, initial_memory_path="", formatted_examples=None):
        assert len(self.memory_examples) == 0
        if initial_memory_path:
            with open(initial_memory_path) as f:
                initial_memory_examples = [json.loads(line)
                                           for line in set(f.read().splitlines())][:]
            initial_memory_examples = self.cl_utils.upstream_data_formatter(initial_memory_examples)
        elif formatted_examples:
            initial_memory_examples = formatted_examples
        for item in initial_memory_examples:
            # Note that we only keep the all answers here now.
            self.memory_examples[item[2]] = (item[0], item[1], item[2])
        self.logger.info(f"Set up the initial memory with {len(self.memory_examples)} examples.")

    def set_up_initial_memory(self, initial_memory_path="", formatted_examples=None):
        raise NotImplementedError

    def load_memory_from_path(self, init_memory_cache_path):
        raise NotImplementedError

    def save_memory_to_path(self, memory_pkl_path):
        raise NotImplementedError

    def retrieve_from_memory(self, query_examples, sample_size, **kwargs):
        raise NotImplementedError

    def store_exampls(self, examples):
        raise NotImplementedError


class RandomMemoryManger(BaseMemoryManager):
    ### Mainly used for ER, MIR

    def __init__(self, logger):
        super().__init__(logger=logger)
        self.logger = logger
        self.name = "random_memory_manager"
        self.memory_examples = {}

    def set_up_initial_memory(self, initial_memory_path="", formatted_examples=None):
        self._load_init_memory_examples(initial_memory_path, formatted_examples)

    def load_memory_from_path(self, init_memory_cache_path):
        with open(init_memory_cache_path, "rb") as f:
            memory_cache = pickle.load(f)
            self.logger.info(f"Load the cache to {f.name}")
        self.memory_examples = memory_cache["memory_examples"]

    def save_memory_to_path(self, memory_pkl_path):
        memory_cache = {}
        memory_cache["memory_examples"] = self.memory_examples
        with open(memory_pkl_path, "wb") as f:
            pickle.dump(memory_cache, f)
            self.logger.info(f"Saved the cache to {f.name}")

    def retrieve_from_memory(self, query_examples=None, sample_size=-1, **kwargs):
        assert sample_size > 0 
        self.logger.info("Randomly retrieve from the memory. `query_examples` not used")
        retrieved_example_ids = random.sample(list(self.memory_examples.keys()), sample_size)
        retrieved_examples = [self.memory_examples[rid] for rid in retrieved_example_ids]
        return retrieved_examples

    def store_examples(self, examples):
        for item in examples:
            # Note that we only keep the all answers here now.
            self.memory_examples[item[2]] = (item[0], item[1], item[2])
        self.logger.info(f"Save {len(examples)} examples to the memory.")


class BartIndexManager(BaseMemoryManager):

    def __init__(self, logger):
        super().__init__(logger=logger)
        self.logger = logger
        self.name = "bart_index_manager"
        self.memory_index = None
        self.memory_examples = {}
        self.bart_model = None
        self.tokenizer = None
        self.cl_utils = ContinualFinetuning(logger=logger)
        self.data_args = None
        self.dim_vector = 2*768
        self.memory_index_sorted_ids = []

    def set_up_data_args(self, args):
        self.data_args = Namespace(
            do_lowercase=args.do_lowercase,
            append_another_bos=args.append_another_bos,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            task_name=args.task_name,
            train_batch_size=args.train_batch_size,
            predict_batch_size=args.predict_batch_size,
        )

    def set_up_initial_memory(self, initial_memory_path="", formatted_examples=None):
        assert self.bart_model is not None
        self._load_init_memory_examples(initial_memory_path, formatted_examples)
        # build index
        initial_memory_example_ids = sorted(list(self.memory_examples.keys()))
        examples = self.get_examples_by_ids(initial_memory_example_ids)
        vectors = self.get_representation(examples)
        self.update_index(initial_memory_example_ids, vectors)

    def update_index(self, example_ids, vectors):
        assert len(example_ids) == len(vectors)
        if not self.memory_index:
            self.memory_index = faiss.IndexFlatL2(self.dim_vector)
        self.memory_index_sorted_ids += example_ids
        vectors = np.array(vectors)
        self.memory_index.add(vectors)

    def set_up_model(self, model, tokenizer):
        del self.bart_model
        del self.tokenizer
        self.bart_model = model
        self.tokenizer = tokenizer

    def get_examples_by_ids(self, example_ids):
        return [self.memory_examples[eid] for eid in example_ids]

    def load_memory_from_path(self, init_memory_cache_path):
        with open(init_memory_cache_path, "rb") as f:
            memory_cache = pickle.load(f)
            self.logger.info(f"Load the cache to {f.name}")
        self.memory_index_sorted_ids = memory_cache["memory_index_sorted_ids"]
        self.memory_index = memory_cache["memory_index"]
        self.memory_examples = memory_cache["memory_examples"]

    def save_memory_to_path(self, memory_pkl_path):
        memory_cache = {}
        memory_cache["memory_index_sorted_ids"] = self.memory_index_sorted_ids
        memory_cache["memory_index"] = self.memory_index
        memory_cache["memory_examples"] = self.memory_examples

        with open(memory_pkl_path, "wb") as f:
            pickle.dump(memory_cache, f)
            self.logger.info(f"Saved the cache to {f.name}")

    def search_index(self, query_vector, k=5):
        D, I = self.memory_index.search(np.array([query_vector]), k)
        retrieved_example_ids = [self.memory_index_sorted_ids[int(eid)] for eid in I[0]]
        return retrieved_example_ids

    def get_query_representation(self, query_examples):
        return self.get_representation(query_examples)

    def retrieve_from_memory(self, query_examples, sample_size, **kwargs):
        input_vectors = self.get_query_representation(query_examples)
        agg_method = kwargs.get("agg_method", "mean")
        rank_method = kwargs.get("rank_method", "most_similar")
        if agg_method == "mean":
            input_vectors = np.array(input_vectors)
            query_vector = np.mean(input_vectors, axis=0)
            if rank_method == "most_different":
                query_vector = -query_vector
            retrieved_example_ids = self.search_index(query_vector, sample_size)
        elif agg_method == "each_topk_then_random":
            each_sample_size = kwargs.get("each_sample_size", 5)
            retrieved_example_ids = []
            for query_vector in input_vectors:
                if rank_method == "most_different":
                    query_vector = -query_vector
                retrieved_example_ids += self.search_index(query_vector, each_sample_size)
            # retrieved_example_ids = set(retrieved_example_ids) # TODO: decide later.
            retrieved_example_ids = random.sample(retrieved_example_ids, sample_size)
        retrieved_examples = self.get_examples_by_ids(retrieved_example_ids)
        return retrieved_examples

    def store_exampls(self, examples):
        example_ids = []
        for item in examples:
            self.memory_examples[item[2]] = item
            example_ids.append(item[2])
        vectors = self.get_representation(examples)
        self.update_index(example_ids, vectors)

    def get_representation(self, examples):
        all_vectors = get_bart_dual_representation(cl_trainer=self.cl_utils, 
                                                    bart_model=self.bart_model, 
                                                    tokenizer=self.tokenizer, 
                                                    data_args=self.data_args, 
                                                    examples=examples)
        return all_vectors

    def load_encoder_model(self, base_model_args):
        self.cl_utils.load_base_model(base_model_args)
        self.set_up_model(model=self.cl_utils.base_model, tokenizer=self.cl_utils.tokenizer)


if __name__ == '__main__':
    from semanticdebugger.debug_algs import run_lifelong_finetune
    parser = run_lifelong_finetune.get_cli_parser()
    args = parser.parse_args()

    debugging_alg, data_args, base_model_args, debugger_args, logger = run_lifelong_finetune.setup_args(
        args)
    args.predict_batch_size = 8
    index_manager = BartIndexManager(logger=logger)
    index_manager.set_up_data_args(args)
    index_manager.load_encoder_model(base_model_args)
    index_manager.initial_memory_path = "exp_results/data_streams/mrqa.nq_train.memory.jsonl"
    # index_manager.set_up_initial_memory()

    # index_manager.save_memory_to_path("exp_results/data_streams/init_memory.pkl")

    index_manager.load_memory_from_path("exp_results/data_streams/init_memory.pkl")

    # sanity check
    query_ids = index_manager.memory_index_sorted_ids[:1]
    print(query_ids)
    retrieved_ids = index_manager.retrieve_from_memory(query_examples=[
                                                       index_manager.memory_examples[qid] for qid in query_ids], sample_size=5, rank_method="most_different")
    print(retrieved_ids)
    for rid in retrieved_ids:
        item = index_manager.memory_examples[rid]
        print("-"*50)
        print(item[2])
        print(item[0])
        print(item[1])
