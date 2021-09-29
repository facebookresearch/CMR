from argparse import Namespace

from numpy.core.defchararray import index
from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
from tqdm import tqdm 
import torch 
from semanticdebugger.models.utils import trim_batch
import json
from semanticdebugger.debug_algs import run_lifelong_finetune

from transformers.modeling_bart import _prepare_bart_decoder_inputs
import numpy as np
import faiss
import pickle 


class IndexManager():

    def __init__(self, logger):
        super().__init__()
        self.logger=logger
        self.name = "index_manager"
        self.memory_index = None
        self.memory_examples = {}
        self.bart_model = None 
        self.tokenizer = None 
        self.cl_utils = ContinualFinetuning(logger=logger)
        self.initial_memory_path = ""
        self.data_args = None 
        self.dim_vector = 2*768
        self.memory_index_sorted_ids = []

    def set_up_data_args(self, args):
        self.data_args = Namespace(
            bug_stream_json_path="",
            pass_pool_jsonl_path="",
            sampled_upstream_json_path="",
            # pass_sample_size=args.pass_sample_size,
            do_lowercase=args.do_lowercase,
            append_another_bos=args.append_another_bos,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            task_name=args.task_name,
            train_batch_size=args.train_batch_size,
            predict_batch_size=args.predict_batch_size,
            num_beams=args.num_beams,
            max_timecode=args.max_timecode,
            accumulate_eval_freq=-1,
            use_sampled_upstream=False,
        )

    def set_up_initial_memory(self):
        assert self.bart_model is not None

        with open(self.initial_memory_path) as f:
            initial_memory_examples = [json.loads(line) for line in set(f.read().splitlines())][:]    # TODO: only for debugging
        initial_memory_examples = self.cl_utils.upstream_data_formatter(initial_memory_examples)
        for item in initial_memory_examples:
            # Note that we only keep the first answer here now.
            self.memory_examples[item[2]] = (item[0], item[1][0:1], item[2])
        self.logger.info(f"Set up the initial memory with {len(self.memory_examples)} examples.")
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

    def retrieve_from_memory(self, query_examples, sample_size=32, rank_method="most_similar"):
        input_vectors = self.get_representation(query_examples)
        input_vectors = np.array(input_vectors)
        query_vector = np.mean(input_vectors, axis=0)
        if rank_method == "most_different":
            query_vector = -query_vector
        retrieved_example_ids = self.search_index(query_vector, sample_size)
        retrieved_examples = [self.memory_examples[rid] for rid in retrieved_example_ids]
        return retrieved_examples


    def store_exampls(self, examples):
        example_ids = []
        for item in examples:
            self.memory_examples[item[2]] = item
            example_ids.append(item[2])
        vectors = self.get_representation(examples)
        self.update_index(example_ids, vectors)


    def get_representation(self, examples):
        data_manager, _ = self.cl_utils.get_dataloader(self.data_args, examples, mode="train", is_training=False)
        all_vectors = [] 
        bart_model = self.bart_model if self.cl_utils.n_gpu ==1 else self.bart_model.module
        bart_model.eval()
        for batch in tqdm(data_manager.dataloader):
            # self.logger.info(f"len(batch)={len(batch)}")
            if self.cl_utils.use_cuda:
                # print(type(batch[0]), batch[0])
                batch = [b.to(torch.device("cuda")) for b in batch]
            pad_token_id = self.tokenizer.pad_token_id
            batch[0], batch[1] = trim_batch(
                batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(
                batch[2], pad_token_id, batch[3])

            ## Encode the input text with BART-encoder
            input_ids = batch[0]
            input_attention_mask = batch[1]
            encoder_outputs = bart_model.model.encoder(
                input_ids, input_attention_mask) 
            x = encoder_outputs[0]
            x = x[:, 0, :]
            input_vectors = x.detach().cpu().numpy()
            

            # self.logger.info(f"input_vectors.shape = {input_vectors.shape}")

            ## Encode the output text with BART-decoder

            output_ids = batch[2]
            output_attention_mask = batch[3]

            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                bart_model.model.config,
                input_ids,
                decoder_input_ids=output_ids,
                decoder_padding_mask=output_attention_mask,
                causal_mask_dtype=bart_model.model.shared.weight.dtype,
            )
            decoder_outputs = bart_model.model.decoder(
                decoder_input_ids,
                encoder_outputs[0],
                input_attention_mask,
                decoder_padding_mask,
                decoder_causal_mask=causal_mask,
                decoder_cached_states=None,
                use_cache=False
            )
            y = decoder_outputs[0]
            y = y[:, 0, :] 
            output_vectors = y.detach().cpu().numpy()
            
            del batch
            del encoder_outputs
            del decoder_outputs
            # self.logger.info(f"output_vectors.shape = {output_vectors.shape}")

            # concatenate the vectors
            vectors = np.concatenate([input_vectors, output_vectors], axis=1)

            # self.logger.info(f"vectors.shape = {vectors.shape}")
            all_vectors += list(vectors)
        return all_vectors

    def load_encoder_model(self, base_model_args):
        self.cl_utils.load_base_model(base_model_args)
        self.set_up_model(model=self.cl_utils.base_model, tokenizer=self.cl_utils.tokenizer)


if __name__ == '__main__':
    parser = run_lifelong_finetune.get_cli_parser()
    args = parser.parse_args()

    debugging_alg, data_args, base_model_args, debugger_args, logger = run_lifelong_finetune.setup_args(
        args)
    args.predict_batch_size = 8
    index_manager = IndexManager(logger=logger)
    index_manager.set_up_data_args(args) 
    index_manager.load_encoder_model(base_model_args)
    index_manager.initial_memory_path = "exp_results/data_streams/mrqa.nq_train.memory.jsonl"
    # index_manager.set_up_initial_memory()

    # index_manager.save_memory_to_path("exp_results/data_streams/init_memory.pkl")

    index_manager.load_memory_from_path("exp_results/data_streams/init_memory.pkl")

    


    # sanity check 
    query_ids = index_manager.memory_index_sorted_ids[:1]
    print(query_ids)
    retrieved_ids = index_manager.retrieve_from_memory(query_examples=[index_manager.memory_examples[qid] for qid in query_ids], sample_size=5, rank_method="most_different")
    print(retrieved_ids)
    for rid in retrieved_ids:
        item = index_manager.memory_examples[rid]
        print("-"*50)
        print(item[2])
        print(item[0])
        print(item[1])
        
