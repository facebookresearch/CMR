import json 

previous = "experiments/eval_data/qa/submission_stream.T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8.json"

current_splits = "experiments/eval_data/qa/submission_stream.T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8-test.json"


with open(previous) as f:
    p = json.load(f)
with open(current_splits) as f:
    splits = json.load(f)

splits.append(p)
 
print(len(splits))

with open(current_splits, "w") as f:
    json.dump(splits, f)