import torch
from tqdm import tqdm
from transformers.modeling_bart import _prepare_bart_decoder_inputs
from transformers.tokenization_utils import trim_batch
import numpy as np
from semanticdebugger.debug_algs.cl_utils import _keep_first_answer


def masked_mean(reps, masks):
    masks = masks.view(reps.size()[0], reps.size()[1], 1) 
    masked_reps = reps * masks 
    masked_reps_sum = masked_reps.sum(dim=1) 
    length_reps = masks.sum(dim=1).view(masked_reps_sum.size()[0], 1)  
    mean_reps = masked_reps_sum / length_reps 
    return mean_reps


def get_bart_dual_representation(cl_trainer, bart_model, tokenizer, data_args, examples, return_all_hidden=False):
    examples_with_single_ans = _keep_first_answer(examples)
    data_manager, _ = cl_trainer.get_dataloader(data_args,
                                                    examples_with_single_ans,
                                                    mode="train",
                                                    is_training=False)
    all_vectors = []
    bart_model = bart_model if cl_trainer.n_gpu == 1 else bart_model.module
    bart_model.eval()
    all_hiddens = {"input_reps":[], "input_masks": [], "output_reps": [] , "output_masks": []}
    for batch in tqdm(data_manager.dataloader):
        # self.logger.info(f"len(batch)={len(batch)}")
        if cl_trainer.use_cuda:
            # print(type(batch[0]), batch[0])
            batch = [b.to(torch.device("cuda")) for b in batch]
        pad_token_id = tokenizer.pad_token_id
        batch[0], batch[1] = trim_batch(
            batch[0], pad_token_id, batch[1])
        batch[2], batch[3] = trim_batch(
            batch[2], pad_token_id, batch[3])

        # Encode the input text with BART-encoder
        input_ids = batch[0]
        input_attention_mask = batch[1]
        encoder_outputs = bart_model.model.encoder(
            input_ids, input_attention_mask)
        x = encoder_outputs[0]
        # x = x[:, 0, :]
        x = masked_mean(x, input_attention_mask)   # use the mean instead of the first

        input_vectors = x.detach().cpu().numpy()

        # self.logger.info(f"input_vectors.shape = {input_vectors.shape}")

        # Encode the output text with BART-decoder

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
        # y = y[:, 0, :]
        y = masked_mean(y, output_attention_mask)   # use the mean instead of the first

        output_vectors = y.detach().cpu().numpy()
 
        # self.logger.info(f"output_vectors.shape = {output_vectors.shape}")

        # concatenate the vectors
        vectors = np.concatenate([input_vectors, output_vectors], axis=1)
        if return_all_hidden:
            all_hiddens["input_reps"] += list(encoder_outputs[0].detach().cpu().numpy())
            all_hiddens["output_reps"] += list(decoder_outputs[0].detach().cpu().numpy())

            all_hiddens["input_masks"] += list(input_attention_mask.detach().cpu().numpy())
            all_hiddens["output_masks"] += list(output_attention_mask.detach().cpu().numpy())
            
            
        # self.logger.info(f"vectors.shape = {vectors.shape}")
        all_vectors += list(vectors)
        
        del batch
        del encoder_outputs
        del decoder_outputs
    if return_all_hidden:    
        return all_hiddens
    else:
        return all_vectors