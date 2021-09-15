from argparse import Namespace
from logging import disable
from semanticdebugger.task_manager.eval_metrics import evaluate_func
from semanticdebugger.models.bart_with_adapater import BartWithAdapterConfig, MyBartWithAdapter
from semanticdebugger.debug_algs.cl_mbcl_alg import KeyValueMemoryModule
from semanticdebugger.models.hypernet import ParameterGenerator
import numpy as np
import torch
from semanticdebugger.models.mybart import MyBart
from semanticdebugger.models import run_bart
from semanticdebugger.models.utils import (convert_model_to_single_gpu,
                                           freeze_embeds, trim_batch)
from semanticdebugger.task_manager.dataloader import GeneralDataset
from transformers import (AdamW, BartConfig, BartTokenizer,
                          get_linear_schedule_with_warmup)

from semanticdebugger.debug_algs.commons import OnlineDebuggingMethod
from semanticdebugger.debug_algs.cl_simple_alg import ContinualFinetuning
from tqdm import tqdm
from torch import log, nn
import torch
from torch.nn import functional as F
import transformers


class HyperBart(nn.Module):
    
    def __init__(self, logger, config):
        super().__init__()
        self.logger = logger
        self.config = config
        self.bart_model = None
        self.weight_generator = None
        self.example_encoder, self.example_tokenizer = None, None
        # self.stored_task_embs = nn.Parameter(torch.zeros(self.config.num_tasks, self.task_emb_dim)) # for Trainable
        # self.register_buffer('stored_task_embs', torch.zeros(self.config.num_tasks, self.task_emb_dim)) # fixed

    def apply_adapter_weights(self, adapter_weights):
        encoder_params, decoder_params = adapter_weights[:self.config.encoder_layers], adapter_weights[self.config.encoder_layers:] 
        
        d_model = self.config.d_model
        d_adapter = self.config.adapter_dim

        for p, encoder_layer in zip(encoder_params, self.bart_model.encoders()):
            # dw, db: down weight, down bias
            # uw, ub: up weight, up bias
            dw, uw, db, ub = p[0:d_model*d_adapter], \
                            p[d_model*d_adapter:d_model*d_adapter*2], \
                            p[d_model*d_adapter*2:d_model*d_adapter*2+d_adapter], \
                            p[d_model*d_adapter*2+d_adapter:d_model*d_adapter*2+d_adapter+d_model]
            encoder_layer.adapter_down_weight = dw.view(d_model, d_adapter)
            encoder_layer.adapter_down_bias = db.view(d_adapter)
            encoder_layer.adapter_up_weight = uw.view(d_adapter, d_model)
            encoder_layer.adapter_up_bias = ub.view(d_model)

            if self.config.adapt_layer_norm:
                encoder_layer.self_attn_layer_norm.weight.data = encoder_layer.self_attn_layer_norm.weight.data + p[-2*d_model: -1*d_model]
                encoder_layer.self_attn_layer_norm.bias.data = encoder_layer.self_attn_layer_norm.bias.data + p[-1*d_model:]


        for p, decoder_layer in zip(decoder_params, self.bart_model.decoders()):
            dw, uw, db, ub = p[0:d_model*d_adapter], \
                            p[d_model*d_adapter:d_model*d_adapter*2], \
                            p[d_model*d_adapter*2:d_model*d_adapter*2+d_adapter], \
                            p[d_model*d_adapter*2+d_adapter:d_model*d_adapter*2+d_adapter+d_model]
            decoder_layer.adapter_down_weight = dw.view(d_model, d_adapter)
            decoder_layer.adapter_down_bias = db.view(d_adapter)
            decoder_layer.adapter_up_weight = uw.view(d_adapter, d_model)
            decoder_layer.adapter_up_bias = ub.view(d_model)

            if self.config.adapt_layer_norm:
                decoder_layer.self_attn_layer_norm.weight.data = decoder_layer.self_attn_layer_norm.weight.data + p[-2*d_model: -1*d_model]
                decoder_layer.self_attn_layer_norm.bias.data = decoder_layer.self_attn_layer_norm.bias.data + p[-1*d_model:]

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False, task_emb=None):
        """"overwrite the bart.forward function"""

        # assert task_emb.dim() == 1
        # generated_weights = None
        generated_weights = self.weight_generator(task_emb.unsqueeze(0))
        # self.bart_model.set_adapter_weights(generated_weights)
        self.apply_adapter_weights(adapter_weights=generated_weights) 
        ret = self.bart_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, is_training=is_training, use_cache=use_cache)
        return ret 


    

    def load_example_encoder(self):
        tmp = KeyValueMemoryModule(self.logger)
        self.example_encoder, self.example_tokenizer = tmp.load_key_encoder(memory_key_encoder=self.config.example_encoder_name)
         
    def get_task_embeddings(self, dataloader):
        # TODO: get the ids of the examples
        # TODO: get the vectors of these ids 
        # TODO: aggreagte the vectors (with mean) to get a task embedding vector.
        examples = dataloader.data
        tmp = KeyValueMemoryModule(self.logger)
        tmp.tokenizer = self.example_tokenizer
        tmp.key_encoder = self.example_encoder
        all_vectors = tmp.encode_examples_for_caching(examples, return_tensors=True)
        all_vectors = torch.stack(all_vectors)
        # print(all_vectors)
        mean_embedding = torch.mean(all_vectors, 0)
        # print(mean_embedding)
        return mean_embedding

    # def init_weight_generator(self):
    #     # make sure config has such attrs
    #         # config.encoder_layers
    #         # config.decoder_layers
    #         # config.activation_function
    #         # config.activation_function
    #         # config.generator_hdim
    #         # config.task_emb_dim
    #         # config.d_model
    #         # config.adapter_dim
    #         # config.adapt_layer_norm
    #     self.weight_generator = ParameterGenerator(self.config)


class HyperCL(ContinualFinetuning):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.name = "hyper_cl"

    def _check_debugger_args(self):
        super()._check_debugger_args()
        required_atts = ["adapter_dim", "example_encoder_name", "task_emb_dim"]
        assert all([hasattr(self.debugger_args, att) for att in required_atts])

    
    def debugger_setup(self, debugger_args):
        self.debugger_args = debugger_args
        self._check_debugger_args()
        model_type, base_model_path = self.base_model_args.model_type, self.base_model_args.base_model_path

        # Set up the additional configs
        config = BartWithAdapterConfig.from_pretrained(model_type)
        config.adapter_dim = debugger_args.adapter_dim
        config.adapt_layer_norm = False # debugger_args.adapt_layer_norm
        # config.unfreeze_hyper_encoder = debugger_args.unfreeze_hyper_encoder
        # config.num_tasks = len(self.all_bug_examples) # number of the overall examples in the error stream.
        config.task_emb_dim = debugger_args.task_emb_dim # the dim of the CLS token embedding of the below model.
        config.example_encoder_name = debugger_args.example_encoder_name


        # Set up the HyperBart model
        self.base_model = HyperBart(self.logger, config)
        hyper_bart = self.base_model    # make an alias to indicate the special arch.
        hyper_bart.bart_model = MyBartWithAdapter(config)
        hyper_bart.weight_generator = ParameterGenerator(config)
        hyper_bart.load_example_encoder()

        # Load the bart model of the HyperBart model.
        self.logger.info(f"Loading checkpoint from {base_model_path} for {model_type} .....")
        mybart_model = MyBart.from_pretrained(model_type, state_dict=convert_model_to_single_gpu(torch.load(base_model_path)))
        hyper_bart.bart_model.model.load_state_dict(mybart_model.model.state_dict(), strict=False)
        

        # TODO: load the cache of both bart and the weight generator 

        if self.use_cuda:
            # Enable multi-gpu training.
            hyper_bart.to(torch.device("cuda"))
            self.logger.info("Moving to the GPUs.")
            if self.n_gpu > 1:
                hyper_bart = torch.nn.DataParallel(hyper_bart)

        hyper_bart = hyper_bart.module if self.n_gpu > 1 else hyper_bart

        # TODO: set up the memory for the "task embedding"
        self.stored_task_embs = None
            ## we can assume that we have a pre-defined number of incoming examples. 
            ## pre-computed by a frozen bert for each example.
            ## set up a method to extend the look-up table.
        
        
        # Set up the optimizer.
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            # Note that we only update the hypernetwork.
            {'params': [p for n, p in hyper_bart.weight_generator.decoders.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': debugger_args.weight_decay},
            {'params': [p for n, p in hyper_bart.weight_generator.decoders.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(self.optimizer_grouped_parameters,
                               lr=debugger_args.learning_rate, eps=debugger_args.adam_epsilon)

        # TODO: double check the decision about warm up for fine-tuning
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=debugger_args.warmup_steps,
                                                         num_training_steps=debugger_args.total_steps)

        self.logger.info(f"Debugger Setup ...... Done!") 

        return 

    
    def load_base_model(self, base_model_args, mode="online_debug"):
        self.base_model_args = base_model_args
        if mode=="offline_eval":
            model_type, base_model_path = self.base_model_args.model_type, self.base_model_args.base_model_path
            # Set up the additional configs
            config = BartWithAdapterConfig.from_pretrained(model_type)
            config.adapter_dim = self.debugger_args.adapter_dim
            config.adapt_layer_norm = False # self.debugger_args.adapt_layer_norm
            # config.unfreeze_hyper_encoder = debugger_args.unfreeze_hyper_encoder
            # config.num_tasks = len(self.all_bug_examples) # number of the overall examples in the error stream.
            config.task_emb_dim = self.debugger_args.task_emb_dim # the dim of the CLS token embedding of the below model.
            config.example_encoder_name = self.debugger_args.example_encoder_name


            # Set up the HyperBart model
            self.base_model = HyperBart(self.logger, config)
        else:
            pass    # the base_model is initiated in the debugger_setup
        return


    def fix_bugs(self, bug_loader, quiet=True):
        # set the states of the hypernetwork and the base model for inference 
        self.base_model.train()

        train_losses = []
        global_step = 0
        pad_token_id = self.tokenizer.pad_token_id
        
        hyper_bart = self.base_model # alias

        task_emb = hyper_bart.get_task_embeddings(bug_loader)

        self.base_model.train()
        train_losses = []
        global_step = 0
        for epoch_id in range(int(self.debugger_args.num_epochs)):
            for batch in tqdm(bug_loader.dataloader, desc=f"Bug-fixing Epoch {epoch_id}", disable=quiet):
                global_step += 1
                # here the batch is a mini batch of the current bug batch
                if self.use_cuda:
                    # print(type(batch[0]), batch[0])
                    batch = [b.to(torch.device("cuda")) for b in batch]
                pad_token_id = self.tokenizer.pad_token_id
                batch[0], batch[1] = trim_batch(
                    batch[0], pad_token_id, batch[1])
                batch[2], batch[3] = trim_batch(
                    batch[2], pad_token_id, batch[3])
                loss = hyper_bart(task_emb=task_emb, input_ids=batch[0], attention_mask=batch[1],
                                       decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                                       is_training=True)
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                train_losses.append(loss.detach().cpu())
                loss.backward()
                self.model_update_steps += 1

                if global_step % self.debugger_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        hyper_bart.parameters(), self.debugger_args.max_grad_norm)
                    self.optimizer.step()    # We have accumulated enough gradients
                    self.scheduler.step()
                    hyper_bart.zero_grad()
        return


    def get_task_split_for_inference(self):
        pass

    def evaluate(self, eval_dataloader=None, verbose=False):
        """Evaluates the performance"""

        if not eval_dataloader:
            eval_dataloader = self.bug_eval_loaders[self.timecode]

        # prepare adapt_dataloaders
        adapt_dataloaders = self.get_adapt_dataloaders(eval_dataloader, verbose=True) 

        predictions = self.base_model_infer_with_adaptation(eval_dataloader, adapt_dataloaders, verbose)
        assert len(predictions) == len(eval_dataloader)
        predictions = [p.strip() for p in predictions]
        results, return_all = evaluate_func(
            predictions, eval_dataloader.data, self.metric, return_all=True)

        return predictions, results, return_all