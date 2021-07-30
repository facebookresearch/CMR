from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
from tqdm import tqdm

mode = "context"
# mode = "question"
num_return_sequences = 3

if mode == "context":
    tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = T5ForConditionalGeneration.from_pretrained(
        "Vamsi/T5_Paraphrase_Paws")
    max_input_length = 888
elif mode == "question":
    model = T5ForConditionalGeneration.from_pretrained(
        'ramsrigouthamg/t5_paraphraser')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    max_input_length = 128


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = model.to(device)


sentence = "when did one child policy end in china ?"
inputs = ["paraphrase: " + sentence + " </s>" for _ in range(128)] 
 
batch_size = 16
num_batches = len(inputs)/batch_size

outputs = []
for batch_inputs in tqdm(np.array_split(inputs, num_batches)):
    tokenized_input = tokenizer.batch_encode_plus(batch_inputs,
                                                    pad_to_max_length=True,
                                                    max_length=max_input_length, return_tensors="pt")

    batch_input_ids, batch_attention_masks = tokenized_input["input_ids"], tokenized_input["attention_mask"]
    batch_input_ids = batch_input_ids.to(device)
    batch_attention_masks = batch_attention_masks.to(device)
    batch_outputs = model.generate(
        input_ids=batch_input_ids, attention_mask=batch_attention_masks,
        max_length=888,
        do_sample=True,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=num_return_sequences
    )
    results = []
    for output in batch_outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
        results.append(line)

    outputs.append(results)

print(outputs)