from argparse import Namespace
from semanticdebugger.debug_algs.cl_utils import _keep_first_answer

from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
from tqdm import tqdm
import torch
from semanticdebugger.debug_algs.index_based.index_manager import BartIndexManager, BaseMemoryManager
from semanticdebugger.debug_algs.index_based.index_utils import get_bart_dual_representation
from semanticdebugger.models.utils import trim_batch
import json


from transformers.modeling_bart import _prepare_bart_decoder_inputs
import numpy as np
import faiss
import pickle
import random
from scipy.spatial import distance

class BartIOIndexManager(BartIndexManager):

    def __init__(self, logger):
        super().__init__(logger=logger)
        self.logger = logger
        self.name = "bart_io_index_manager"
        self.memory_index = {"input": None, "output": None} 
        self.dim_vector = 768 

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


    def update_index(self, example_ids, vectors):
        assert len(example_ids) == len(vectors)
        if not self.memory_index["input"]:
            self.memory_index["input"] = faiss.IndexFlatL2(self.dim_vector)
            self.memory_index["output"] = faiss.IndexFlatL2(self.dim_vector)
        self.memory_index_sorted_ids += example_ids
        ## add to input
        input_vectors = np.array([v[:self.dim_vector] for v in vectors])
        self.memory_index["input"].add(input_vectors)
        ## add to output
        output_vectors = np.array([v[self.dim_vector:] for v in vectors])
        self.memory_index["output"].add(output_vectors)
    

    def search_index(self, query_vector, k=5, partition="input", return_index_ids=False):
        if partition=="input":
            query_vector = query_vector[:self.dim_vector]
        elif partition=="output":
            query_vector = query_vector[self.dim_vector:]
        D, I = self.memory_index[partition].search(np.array([query_vector]), k) 
        scores = D[0]
        if return_index_ids:
            return I[0]
        else:
            retrieved_example_ids = [self.memory_index_sorted_ids[int(eid)] for eid in I[0]]
            return retrieved_example_ids
 
    def retrieve_from_memory(self, query_examples, sample_size, **kwargs):
        input_vectors = self.get_query_representation(query_examples)
        agg_method = kwargs.get("agg_method", "each_topk_then_random")
        rank_method = kwargs.get("rank_method", "most_similar")
        if agg_method == "mean":
            # input_vectors = np.array(input_vectors)
            # query_vector = np.mean(input_vectors, axis=0)
            # if rank_method == "most_different":
            #     # query_vector = -query_vector
            #     pass
            # retrieved_example_ids = self.search_index(query_vector, sample_size)
            pass
        elif agg_method == "each_topk_then_random":
            each_sample_size = kwargs.get("each_sample_size", 5)
            each_sim_sample_size = kwargs.get("each_sample_size", 30)
            retrieved_example_ids = []
            for query_vector in input_vectors: 
                sim_input_index_ids = self.search_index(query_vector, each_sim_sample_size, partition="input", return_index_ids=True)
                sim_output_vectors = [self.memory_index["output"].reconstruct(int(eid)) for eid in sim_input_index_ids]
                query_output_vector = query_vector[self.dim_vector:]
                distances = [distance.cosine(query_output_vector, s) for s in sim_output_vectors]
                retrieved_ids = [int(x) for _, x in sorted(zip(distances, sim_input_index_ids), reverse=True)]
                retrieved_example_ids += [self.memory_index_sorted_ids[int(eid)] for eid in retrieved_ids][:each_sample_size]
        retrieved_examples = self.get_examples_by_ids(retrieved_example_ids)
        retrieved_examples = random.sample(retrieved_examples, sample_size) # TODO: consider ranking 
        return retrieved_examples

if __name__ == '__main__':
    from semanticdebugger.debug_algs import run_lifelong_finetune
    parser = run_lifelong_finetune.get_cli_parser()
    args = parser.parse_args()

    debugging_alg, data_args, base_model_args, debugger_args, logger = run_lifelong_finetune.setup_args(
        args)
    args.predict_batch_size = 8
    index_manager = BartIOIndexManager(logger=logger)
    index_manager.set_up_data_args(args)
    index_manager.load_encoder_model(base_model_args)
    index_manager.initial_memory_path = "exp_results/data_streams/mrqa.nq_train.memory.jsonl"
    # index_manager.initial_memory_path = "data/mrqa_naturalquestions/mrqa_naturalquestions_train.jsonl"
    index_manager.set_up_initial_memory(index_manager.initial_memory_path, cut_off=None)

    index_manager.save_memory_to_path("exp_results/data_streams/bart_io_index.init_memory.pkl")

    # item = [0,0,0]
    # item[2] = "mrqa_naturalquestions-train-10-copy"
    # item[0] = "Context: The movie was shot in LA and the Hawaiian islands of Bologno and Kologno between March 3 , 2020 and May 25 , 2020 . The movie is deliberately vague about which Hawaiian island its latter portion depicts ; thus , the characters hike across a rope bridge on Bologno and arrive in the next scene at a spectacular waterfall on Kologno , rather than the ordinary irrigation dam and pond on Bologno where the actual trail terminates . | Question: which did they hike in just go with it ?"
    # item[1] = ['xxa xzcvxzcv q234er2314', 'adsfasdf sad fgcxbv dsafv adsf .']
    
    # index_manager.store_examples([item])


    # query_ids = ["mrqa_naturalquestions-train-10"]
    # print(index_manager.memory_examples[query_ids[0]])
    # retrieved_exmaples = index_manager.retrieve_from_memory(query_examples=[
    #                                                    index_manager.memory_examples[qid] for qid in query_ids], each_sample_size=10, sample_size=10, rank_method="")
    
    # for item in retrieved_exmaples: 
    #     print("-"*50)
    #     print(item[2])
    #     print(item[0])
    #     print(item[1])