import json
import argparse
import pandas as pd
from argparse import Namespace
import numpy as np
import glob, os

from pandas.core import base

def show_result(path): 
    o = json.load(open(path))
    r = {}
    debugger_args = eval(o["debugger_args"])
    data_args = eval(o["data_args"])

    r["path"] = path
    # r["prefix"] = prefix
    r["method_class"] = o["method_class"]
    r["cl_method"] = o["method_class"]
    if r["cl_method"] == "online_ewc":
        ewc_lambda= debugger_args.ewc_lambda
        r["cl_method"] = f'{r["cl_method"]}-{ewc_lambda}'
    elif r["cl_method"] == "er":
        replay_size = debugger_args.replay_size
        replay_freq = debugger_args.replay_frequency
        r["cl_method"] = f'{r["cl_method"]}-{replay_size}-{replay_freq}'
    elif r["cl_method"] == "mir":
        replay_size = debugger_args.replay_size
        replay_freq = debugger_args.replay_frequency
        replay_candidate_size = debugger_args.replay_candidate_size
        r["cl_method"] = f'{r["cl_method"]}-{replay_size}/{replay_candidate_size}-{replay_freq}'
        # replay_size = debugger_args.replay_size
    r["steps"] = o["model_update_steps"]
    
    r["lr"] = 0 if r["cl_method"]=="none_cl" else debugger_args.learning_rate
    r["num_epochs"] = 0 if r["cl_method"]=="none_cl" else debugger_args.num_epochs
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
    if len(EFRs) != ns_config["T"]:
        return None
    last_step = online[-1] 
    assert last_step["timecode"] == ns_config["T"] -1 
    r["CSR(T)"] = last_step["CSR"]
    r["AEFR(T)"] = float(np.mean(EFRs))
    # print(len(UKRs))
    r["UKR(T)"] = UKRs[-1]
    r["OKR(T)"] = OKRs[-1]
    r["KG(T)"] = KGs[-1]
    return r
    


os.chdir("experiments/results/qa/")
os.makedirs("csvs/", exist_ok=True)
result_files = []
for file in glob.glob("*.json"):
    result_files.append(file)

results = []
for r_file in result_files:
    r = show_result(r_file)
    if r:
        results.append(r)

results.sort(key=lambda x:x["cl_method"])

results = pd.DataFrame(results)

pd.set_option('display.float_format', lambda x: '%.5f' % x)

for ns_config in results.ns_config.unique():
    # print(ns_config)
    r = results[results["ns_config"]==ns_config]
    # r = r[(r["AEFR(T)"]>0.85) | (r["cl_method"]=="none_cl")]
    
    def _sort(column):
        # def tm_sorter(column):
        """Sort function"""
        cl_methods = ['none_cl', "simple_cl", "online_ewc", "er", "mir"]
        correspondence = {team: order for order, team in enumerate(cl_methods)}
        return column.map(correspondence)
    r = r.sort_values(by=["steps", "lr", "num_epochs", "cl_method"])
    r = r.sort_values(by="method_class", key=_sort, kind="mergesort")
    r = r.drop(columns=["path", "ns_config", "method_class"])
    r.to_csv(f"csvs/{ns_config}.csv", index=False)
    print("-"*50)
    print(f'ns_config="{ns_config.replace(",", " & ")}",')
    print(open(f"csvs/{ns_config}.csv").read())



    # find the baseline of simple_cl for er and mir 
    # saved_baseline = {}
    # for index, row in r.iterrows():
    #     lr = row["lr"]; ne = row["num_epochs"]
    #     prefix = f"lr={lr}_ne={ne}"
    #     if row["cl_method"] == "simple_cl":
    #         saved_baseline[prefix] = row
    
    # # TOOD: to show the relative performance gain
    # for index, row in r.iterrows():
    #     # print(row)
    #     lr = row["lr"]; ne = row["num_epochs"]
    #     prefix = f"lr={lr}_ne={ne}"
    #     if row["cl_method"] not in ["none_cl", "simple_cl"]:
    #         # compute the relative scores 
    #         base_scores = saved_baseline[prefix]
    #         row["stpes"] = (row["stpes"] - base_scores["stpes"])
    #         row["CSR(T)"] = (row["CSR(T)"] - base_scores["CSR(T)"])
    #         row["AEFR(T)"] = (row["AEFR(T)"] - base_scores["AEFR(T)"])
    #         row["UKR(T)"] = (row["UKR(T)"] - base_scores["UKR(T)"])
    #         row["OKR(T)"] = (row["OKR(T)"] - base_scores["OKR(T)"])
    #         row["KG(T)"] = (row["KG(T)"] - base_scores["KG(T)"])
    #         row["cl_method"] = row["cl_method"] + "*"
    #         # print(",".join(row.to_csv(header=False, index=False).split('\n')))
    #     # print(row)

    