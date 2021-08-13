import json
import argparse
from re import S
from semanticdebugger.models.utils import set_seeds
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np 
from tqdm import tqdm
import spacy, nltk

parser = argparse.ArgumentParser()

parser.add_argument(
        "--data_stream_path",
        default="exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.wr.json", type=str)
parser.add_argument(
        "--data_paraphrased_dict",
        default="exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.wr.para_data.json", type=str)

parser.add_argument('--num_shards', type=int, default=4)
parser.add_argument('--shard_id', type=int, default=0)

def get_duplicate_ids(data_stream):
    seen_ids = set()
    examples_to_paraphrase = {}
    for episode in data_stream:
        for item in episode:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
            else:
                examples_to_paraphrase[item["id"]] = item 
    return examples_to_paraphrase



def split_sentences(text): 
    nlp = spacy.load('en_core_web_sm')  # python -m spacy download en_core_web_sm
    # text = "How are you today? I hope you have a great day"
    docs = nlp(text)
    sents = []
    for sent in docs.sents:
        sents.append(str(sent).strip())
    return sents


def inference(tokenizer, model, inputs, K=5, max_input_length=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device: ", device)
    inputs = add_prompt(inputs)
    tokenized_input = tokenizer.batch_encode_plus(inputs,
                                                    pad_to_max_length=True,
                                                    max_length=max_input_length, return_tensors="pt")

    batch_input_ids, batch_attention_masks = tokenized_input["input_ids"], tokenized_input["attention_mask"]
    batch_input_ids = batch_input_ids.to(device)
    batch_attention_masks = batch_attention_masks.to(device)
    batch_outputs = model.generate(
        input_ids=batch_input_ids, attention_mask=batch_attention_masks,
        max_length=max_input_length,
        do_sample=True,
        top_k=120,
        top_p=0.93,
        early_stopping=True,
        num_return_sequences=K
    )
    results = []
    for output in batch_outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
        results.append(line)
    splitted_results = np.array_split(results, len(inputs))
    assert len(splitted_results[0]) == K
    for id in range(len(splitted_results)):
        splitted_results[id] = list(splitted_results[id])
    splitted_results = list(splitted_results)

    def is_too_similar(s1, s2):
        return nltk.edit_distance(s1.replace(" ", ""), s2.replace(" ", "")) <= 10

    for id in range(len(inputs)):
        s = inputs[id]
        splitted_results[id] = [p for p in splitted_results[id] if not is_too_similar(p, s)]
    return splitted_results


def add_prompt(sentences):
    return [f"paraphrase: {s} </s>" for s in sentences]

def get_paraphrased_example(model, tokenizer, example):
    context, question = example["input"].split("|")
    context = context.replace("Context: ", "")
    question = question.replace("Question: ", "")
    context_sentences = split_sentences(context)

    context_paraphrases = inference(tokenizer, model, context_sentences, K=3)
    question_paraphrases = inference(tokenizer, model, [question], K=3)
    return context_sentences, context_paraphrases, question_paraphrases
    # print(len(sentences), len(paraphrases), len(paraphrases[0]))
    # print(sentences)
    # print(paraphrases)



def init_para_model():
    tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = T5ForConditionalGeneration.from_pretrained(
        "Vamsi/T5_Paraphrase_Paws")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model = model.to(device)
    return model, tokenizer


if __name__ == '__main__':

    # init the paraphrasing model.
    
    set_seeds(42)

    args = parser.parse_args()
    with open(args.data_stream_path, "r") as f :
        data_stream = json.load(f)
    
    examples_to_paraphrase = get_duplicate_ids(data_stream)
    print(len(examples_to_paraphrase))

    model, tokenizer = init_para_model()

    paraphrased_examples = {}

    all_ids = sorted(list(examples_to_paraphrase.keys()))

    current_ids = np.array_split(all_ids, args.num_shards)[args.shard_id]

    for _id in tqdm(current_ids, desc=f"shard_id: {args.shard_id}"):
        example = examples_to_paraphrase[_id]
        context_sentences, context_paraphrases, question_paraphrases = get_paraphrased_example(model, tokenizer, example)
        example["context_sentences"] = context_sentences
        example["context_paraphrases"] = context_paraphrases
        example["question_paraphrases"] = question_paraphrases
        paraphrased_examples[_id] = example
    
    with open(args.data_paraphrased_dict, "w") as f :
        json.dump(paraphrased_examples, f)

        
"""
n_threads=16
n_gpus=8 
start_gpuid=0
for (( thread=0; thread<${n_threads}; thread++ ))
do
    gpu=$(($start_gpuid + $thread % n_gpus))
    echo $thread, $gpu
    CUDA_VISIBLE_DEVICES=${gpu} python semanticdebugger/benchmark_gen/para_stream.py \
        --data_paraphrased_dict "exp_results/data_streams/paraphrase/mrqa_naturalquestions_dev.data_stream.test.wr.para_data_${thread}.json" \
        --num_shards $n_threads --shard_id ${thread} &
done


python semanticdebugger/benchmark_gen/merge_json_file.py \
    --input_file_pattern exp_results/data_streams/paraphrase/mrqa_naturalquestions_dev.data_stream.test.wr.para_data_#.json \
    --range "range(16)" \
    --output_file exp_results/data_streams/paraphrase/mrqa_naturalquestions_dev.data_stream.test.wr.para_data.json
"""
