import json
from tqdm import tqdm

preprocessed_path = "data/mrqa_squad/mrqa_squad_train-BartTokenized.json"

with open(preprocessed_path, "r") as f:
    print("Loading", preprocessed_path)
    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
        metadata = json.load(f)

print(len(metadata))

for data_index, data_item in tqdm(enumerate(metadata)):
    print(data_index)
    x, y = data_item
    assert x<y

print()