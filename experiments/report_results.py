import json
import argparse
import pandas as pd
from argparse import Namespace
import numpy as np

def show_result(path): 
    o = json.load(open(path))
    r = {}
    r["path"] = path
    # r["prefix"] = prefix
    r["cl_method"] = o["method_class"]
    r["model_update_steps"] = o["model_update_steps"]
    debugger_args = eval(o["debugger_args"])
    data_args = eval(o["data_args"])
    r["lr"] = debugger_args.learning_rate
    r["num_epochs"] = debugger_args.num_epochs
    start = data_args.submission_stream_data.index("submission_stream.") + len("submission_stream.")
    end = data_args.submission_stream_data.index(".json") 
    ns_config_str = data_args.submission_stream_data[start:end]
    r["ns_config"] = ns_config_str
    ns_config = eval(f"dict({ns_config_str})")
    # r.update(ns_config)


    online = o["online_eval_results"]
    EFRs = [item["EFR"] for item in online]
    UKRs = [item["UKR"] for item in online if "UKR" in item]
    OKRs = [item["OKR"] for item in online if "OKR" in item]
    KGs = [item["KG"] for item in online if "KG" in item]
    assert len(EFRs) == ns_config["T"]
    last_step = online[-1] 
    assert last_step["timecode"] == ns_config["T"] -1 
    r["CSR(T)"] = last_step["CSR"]
    r["AEFR(T)"] = float(np.mean(EFRs))
    r["UKR(T-4)"] = UKRs[-1]
    r["OKR'(T-4)"] = OKRs[-1]
    r["KG(T-4)"] = KGs[-1]
    return r
    


import glob, os
os.chdir("experiments/results/qa/")
result_files = []
for file in glob.glob("*.json"):
    result_files.append(file)

results = []
for r_file in result_files:
    results.append(show_result(r_file))

results.sort(key=lambda x:x["cl_method"])

results = pd.DataFrame(results)
print(results)