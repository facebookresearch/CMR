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
            initial_memory_examples = [json.loads(line) for line in set(f.read().splitlines())][:500]    # TODO: only for debugging
        initial_memory_examples = self.cl_utils.upstream_data_formatter(initial_memory_examples)
        for item in initial_memory_examples:
            # Note that we only keep the first answer here now.
            self.memory_examples[item[2]] = (item[0], item[1][0:1], item[2])
        self.logger.info(f"Set up the initial memory with {len(self.memory_examples)} examples.")
        initial_memory_example_ids = sorted(list(self.memory_examples.keys()))
        self.memory_index_sorted_ids = initial_memory_example_ids
        vectors = self.get_representation(initial_memory_example_ids)
        self.update_index(initial_memory_example_ids, vectors)


    def update_index(self, example_ids, vectors):
        assert len(example_ids) == len(vectors)
        if not self.memory_index:
            self.init_index(vectors)
        else:
            # TODO: update faiss index?
            pass 


    def init_index(self, vectors):        
        self.memory_index = faiss.IndexFlatL2(self.dim_vector)
        vectors = np.array(vectors)
        self.memory_index.add(vectors)

        
    def query_examples(self, query_vectors, k=5):
        """
        Returns samples from buffer using K-nearest neighbour approach
        """
        retrieved_examples = []
        for query_vector in query_vectors:
            D, I = self.memory_index.search(np.array([query_vector]), k)
            retrieved_examples.append([self.memory_index_sorted_ids[int(eid)] for eid in I[0]])
        return retrieved_examples

    def set_up_model(self, model, tokenizer):
        self.bart_model = model 
        self.tokenizer = tokenizer

    def get_examples_by_ids(self, example_ids):
        return [self.memory_examples[eid] for eid in example_ids]


    

    def get_representation(self, example_ids):
        examples = self.get_examples_by_ids(example_ids) 
        
        data_manager, _ = self.cl_utils.get_dataloader(self.data_args, examples, mode="train", is_training=False)
        all_vectors = []
        bart_model = self.bart_model.module
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



if __name__ == '__main__':
    parser = run_lifelong_finetune.get_cli_parser()
    args = parser.parse_args()

    debugging_alg, data_args, base_model_args, debugger_args, logger = run_lifelong_finetune.setup_args(
        args)
    args.predict_batch_size = 8
    index_manager = IndexManager(logger=logger)
    index_manager.set_up_data_args(args) 
    index_manager.cl_utils.load_base_model(base_model_args)
    index_manager.set_up_model(model=index_manager.cl_utils.base_model, tokenizer=index_manager.cl_utils.tokenizer)
    index_manager.initial_memory_path = "exp_results/data_streams/mrqa.nq_train.memory.jsonl"
    
    index_manager.set_up_initial_memory()

    query_ids = list(index_manager.memory_examples.keys())[:3]
    tensors = index_manager.get_representation(example_ids=query_ids)
    print(query_ids, tensors)
    retrived_ids = index_manager.query_examples(tensors)
    print(retrived_ids)
