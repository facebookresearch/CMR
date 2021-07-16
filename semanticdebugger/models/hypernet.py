import torch
import torch.nn as nn
from transformers import BartModel, RobertaModel
from transformers.activations import ACT2FN

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=0.0000001)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

def RegularLinear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=1)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
    
def HNetLinear(config, in_features, out_features, input_dim, output_dim, bias=True):
    var_e = 2 / (config.task_emb_dim + config.long_term_task_emb_num)
    weight_var_fanin = 1 / (2 * in_features * input_dim * var_e)
    weight_var_fanout = 1 / (in_features * output_dim * var_e)
    bias_var_fanin = 1 / (2 * config.task_emb_dim * var_e)
    bias_var_fanout = max((1 - (input_dim / output_dim)) / (config.task_emb_dim * var_e), 1e-10)
    weight_var = 2 / (1 / weight_var_fanin + 1 / weight_var_fanout)
    bias_var = 2 / (1 / bias_var_fanin + 1 / bias_var_fanout)

    m = nn.Linear(in_features, out_features, bias)
    nn.init.normal_(m.weight, 0, weight_var ** 0.5)
    if bias:
        nn.init.normal_(m.bias, 0, bias_var ** 0.5)
    return m


class WeightGenerator_MLP(nn.Module):
    # takes in a encoded task description and generates parameters of an adapter
    def __init__(self, config):
        super().__init__()

        self.input_dim = 768 # config.d_model
        self.hidden_dim = config.generator_hdim
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

        self.encoder = RobertaModel.from_pretrained('roberta-base')
        if self.config.unfreeze_hyper_encoder:
            self.encoder.train()
        else:
            self.encoder.eval()

        self.decoders = nn.ModuleList([
            WeightGenerator_MLP(config) for _ in range(config.encoder_layers + config.decoder_layers)
        ])

    def encode(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):

        # to save memory, the encoder (bart) here is frozen
        if self.config.unfreeze_hyper_encoder:
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
            )
        else:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids,
                    attention_mask=attention_mask,
                )

        x = outputs[0] # last hidden state
        x = x[:, 0, :] # take <s> token (equiv. to [CLS])

        # eos_mask = input_ids.eq(self.config.eos_token_id)
        # if len(torch.unique(eos_mask.sum(1))) > 1:
        #     raise ValueError("All examples must have the same number of <eos> tokens.")
        # sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        
        # return sentence_representation
        return x

    def decode(self, sr):
        return [one_decoder(sr) for one_decoder in self.decoders]

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):
        
        h = self.encode(input_ids, attention_mask, encoder_outputs, decoder_input_ids,
                        decoder_attention_mask, decoder_cached_states, use_cache, is_training)

        params = self.decode(h)

        return params

class GrowingBart(nn.Module):
    def __init__(self, model, meta_model, config):
        super().__init__()

        self.config = config
        self.model = model
        self.meta_model = meta_model

    def set_relation(self, rel_ids, rel_masks):
        # generate adapter parameters using task descriptions
        generated_params = self.meta_model(rel_ids, attention_mask=rel_masks)

        # apply the parameters to the adapters
        self.apply_params_to_adapters(generated_params)

    def forward(self, rel_ids, rel_masks, input_ids, input_masks, output_ids, output_masks, is_training=False):
        # generate adapter parameters using task descriptions
        generated_params = self.meta_model(rel_ids, attention_mask=rel_masks)

        # apply the parameters to the adapters
        self.apply_params_to_adapters(generated_params)
        
        # use the adapted model to make zero-shot inference
        ret = self.model(input_ids, attention_mask=input_masks,
                    decoder_input_ids=output_ids,
                    decoder_attention_mask=output_masks,
                    is_training=is_training
        )

        return ret       

    def apply_params_to_adapters(self, generated_params):
        encoder_params, decoder_params = generated_params[:self.config.encoder_layers], generated_params[self.config.encoder_layers:] 
        
        d_model = self.config.d_model
        d_adapter = self.config.adapter_dim

        for p, encoder_layer in zip(encoder_params, self.model.encoders()):
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


        for p, decoder_layer in zip(decoder_params, self.model.decoders()):
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

        # a = self.model.decoders()[-4]
        # print(a.adapter_down_weight)
        # print(a.adapter_down_bias)
        # print(a.adapter_up_weight)
        # print(a.adapter_up_bias)
        
