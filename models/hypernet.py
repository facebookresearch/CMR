# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from transformers import BartModel, RobertaModel
from transformers.activations import ACT2FN
from typing import List

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=0.0000001)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

# def RegularLinear(in_features, out_features, bias=True):
#     m = nn.Linear(in_features, out_features, bias)
#     nn.init.xavier_uniform_(m.weight, gain=1)
#     if bias:
#         nn.init.constant_(m.bias, 0.0)
#     return m
    
# def HNetLinear(config, in_features, out_features, input_dim, output_dim, bias=True):
#     var_e = 2 / (config.task_emb_dim + config.long_term_task_emb_num)
#     weight_var_fanin = 1 / (2 * in_features * input_dim * var_e)
#     weight_var_fanout = 1 / (in_features * output_dim * var_e)
#     bias_var_fanin = 1 / (2 * config.task_emb_dim * var_e)
#     bias_var_fanout = max((1 - (input_dim / output_dim)) / (config.task_emb_dim * var_e), 1e-10)
#     weight_var = 2 / (1 / weight_var_fanin + 1 / weight_var_fanout)
#     bias_var = 2 / (1 / bias_var_fanin + 1 / bias_var_fanout)

#     m = nn.Linear(in_features, out_features, bias)
#     nn.init.normal_(m.weight, 0, weight_var ** 0.5)
#     if bias:
#         nn.init.normal_(m.bias, 0, bias_var ** 0.5)
#     return m


class MLP_Task2Adapter(nn.Module):
    # takes in a encoded task description and generates parameters of an adapter
    def __init__(self, config):
        super().__init__()

        self.input_dim = config.task_emb_dim # 768?
        self.hidden_dim = config.generator_hdim
        # TODO: set this output_dim = # params of adapters automatically.
        self.output_dim = config.d_model * config.adapter_dim * 2 + config.d_model + config.adapter_dim
        if config.adapt_layer_norm:
            self.output_dim += 2 * config.d_model

        self.linear1 = Linear(self.input_dim, self.hidden_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        self.linear2 = Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x.view(-1)


class ParameterGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config 

        modules = []
        num_adapters = config.encoder_layers + config.decoder_layers # int
        for _ in range(num_adapters):
            modules.append(MLP_Task2Adapter(config))
        self.decoders = nn.ModuleList(modules)

    def decode(self, task_emb):
        return [d(task_emb) for d in self.decoders]

    def forward(self, task_embedding, concat=False):
        adapter_params = self.decode(task_embedding)
        if concat:
            adapter_params = torch.cat(adapter_params)
        return adapter_params 


# class GrowingBart(nn.Module):
#     def __init__(self, model, meta_model, config):
#         super().__init__()

#         self.config = config
#         self.model = model
#         self.meta_model = meta_model

#     def set_relation(self, rel_ids, rel_masks):
#         # generate adapter parameters using task descriptions
#         generated_params = self.meta_model(rel_ids, attention_mask=rel_masks)

#         # apply the parameters to the adapters
#         self.apply_params_to_adapters(generated_params)

#     def forward(self, rel_ids, rel_masks, input_ids, input_masks, output_ids, output_masks, is_training=False):
#         # generate adapter parameters using task descriptions
#         generated_params = self.meta_model(rel_ids, attention_mask=rel_masks)

#         # apply the parameters to the adapters
#         self.apply_params_to_adapters(generated_params)
        
#         # use the adapted model to make zero-shot inference
#         ret = self.model(input_ids, attention_mask=input_masks,
#                     decoder_input_ids=output_ids,
#                     decoder_attention_mask=output_masks,
#                     is_training=is_training
#         )

#         return ret       

#     def apply_params_to_adapters(self, generated_params):
#         encoder_params, decoder_params = generated_params[:self.config.encoder_layers], generated_params[self.config.encoder_layers:] 
        
#         d_model = self.config.d_model
#         d_adapter = self.config.adapter_dim

#         for p, encoder_layer in zip(encoder_params, self.model.encoders()):
#             # dw, db: down weight, down bias
#             # uw, ub: up weight, up bias
#             dw, uw, db, ub = p[0:d_model*d_adapter], \
#                             p[d_model*d_adapter:d_model*d_adapter*2], \
#                             p[d_model*d_adapter*2:d_model*d_adapter*2+d_adapter], \
#                             p[d_model*d_adapter*2+d_adapter:d_model*d_adapter*2+d_adapter+d_model]
#             encoder_layer.adapter_down_weight = dw.view(d_model, d_adapter)
#             encoder_layer.adapter_down_bias = db.view(d_adapter)
#             encoder_layer.adapter_up_weight = uw.view(d_adapter, d_model)
#             encoder_layer.adapter_up_bias = ub.view(d_model)

#             if self.config.adapt_layer_norm:
#                 encoder_layer.self_attn_layer_norm.weight.data = encoder_layer.self_attn_layer_norm.weight.data + p[-2*d_model: -1*d_model]
#                 encoder_layer.self_attn_layer_norm.bias.data = encoder_layer.self_attn_layer_norm.bias.data + p[-1*d_model:]


#         for p, decoder_layer in zip(decoder_params, self.model.decoders()):
#             dw, uw, db, ub = p[0:d_model*d_adapter], \
#                             p[d_model*d_adapter:d_model*d_adapter*2], \
#                             p[d_model*d_adapter*2:d_model*d_adapter*2+d_adapter], \
#                             p[d_model*d_adapter*2+d_adapter:d_model*d_adapter*2+d_adapter+d_model]
#             decoder_layer.adapter_down_weight = dw.view(d_model, d_adapter)
#             decoder_layer.adapter_down_bias = db.view(d_adapter)
#             decoder_layer.adapter_up_weight = uw.view(d_adapter, d_model)
#             decoder_layer.adapter_up_bias = ub.view(d_model)

#             if self.config.adapt_layer_norm:
#                 decoder_layer.self_attn_layer_norm.weight.data = decoder_layer.self_attn_layer_norm.weight.data + p[-2*d_model: -1*d_model]
#                 decoder_layer.self_attn_layer_norm.bias.data = decoder_layer.self_attn_layer_norm.bias.data + p[-1*d_model:]

#         # a = self.model.decoders()[-4]
#         # print(a.adapter_down_weight)
#         # print(a.adapter_down_bias)
#         # print(a.adapter_up_weight)
#         # print(a.adapter_up_bias)
        
