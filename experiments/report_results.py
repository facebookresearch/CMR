import json
import argparse
import pandas as pd
from argparse import Namespace
import numpy as np
import glob, os

from pandas.core import base

def ema(values, period):
    values = pd.DataFrame(np.array(values))
    emas = pd.Series.ewm(values, span=period).mean()
    return emas[0][len(values)-1]

def sma(values, period):
    return float(np.mean(values))
    # values = pd.DataFrame(np.array(values))
    # emas = pd.Series.ewm(values, span=period).mean()
    # return emas[0][len(values)-1]


def show_result(path): 
    # if path == "qa_nonecl_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_offline=yes_result.json":
    #     print()
    o = json.load(open(path))
    r = {}
    debugger_args = eval(o["debugger_args"])
    data_args = eval(o["data_args"])

    r["path"] = path.replace(",", "|")
    # r["prefix"] = prefix
    r["method_class"] = o["method_class"]
    r["cl_method"] = o["method_class"]
    if r["cl_method"] == "simple_cl":
        if hasattr(debugger_args, "diff_loss_weight"):
            r["cl_method"] = f'{r["cl_method"]}-l2w={debugger_args.diff_loss_weight}'
    elif r["cl_method"] == "online_ewc":
        ewc_lambda= debugger_args.ewc_lambda
        ewc_gamma= debugger_args.ewc_gamma
        r["cl_method"] = f'{r["cl_method"]}-{ewc_lambda}-{ewc_gamma}'
    elif r["cl_method"] == "er":
        replay_size = debugger_args.replay_size
        replay_freq = debugger_args.replay_frequency
        r["cl_method"] = f'{r["cl_method"]}-{replay_size}-{replay_freq}'
        if hasattr(debugger_args, "diff_loss_weight"):
            r["cl_method"] = f'{r["cl_method"]}-l2w={debugger_args.diff_loss_weight}'
    elif r["cl_method"] == "mir":
        replay_size = debugger_args.replay_size
        replay_freq = debugger_args.replay_frequency
        replay_candidate_size = debugger_args.replay_candidate_size
        mir_abalation_args = debugger_args.mir_abalation_args
        r["cl_method"] = f'{r["cl_method"]}-{replay_size}/{replay_candidate_size}-{replay_freq}-{mir_abalation_args}'
        if hasattr(debugger_args, "diff_loss_weight"):
            r["cl_method"] = f'{r["cl_method"]}-l2w={debugger_args.diff_loss_weight}'
        # replay_size = debugger_args.replay_size
    elif r["cl_method"] == "index_cl_bart_io_index":
        replay_size = debugger_args.replay_size
        replay_freq = debugger_args.replay_frequency
        r["cl_method"] = f'{r["cl_method"]}-{replay_size}-{replay_freq}'
        
    r["steps"] = o["model_update_steps"]
    
    r["lr"] = 0 if r["cl_method"]=="none_cl" else debugger_args.learning_rate
    r["num_epochs"] = 0 if r["cl_method"]=="none_cl" else debugger_args.num_epochs
    start = data_args.submission_stream_data.index("submission_stream.") + len("submission_stream.")
    end = data_args.submission_stream_data.index(".json") 
    ns_config_str = data_args.submission_stream_data[start:end]
    r["ns_config"] = ns_config_str
    ns_config = eval(f"dict({ns_config_str})")
    r.update(ns_config)
    
    if ns_config["b"] == 128:
        print()

    online = o["online_eval_results"]
    EFRs = [item["EFR"] for item in online]
    UKRs = [item["UKR"] for item in online if "UKR" in item]
    OKRs = [item["OKR"] for item in online if "OKR" in item]
    KGs = [item["KG"] for item in online if "KG" in item]
    CSRs = [item["CSR"] for item in online if "CSR" in item]
    if len(EFRs) != ns_config["T"]:
        return None
    last_step = online[-1] 
    assert last_step["timecode"] == ns_config["T"] -1 
    
    
    r["AEFR(T)"] = float(np.mean(EFRs))

    r["AUKR"] = sma(UKRs, 3)
    r["AOKR"] = sma(OKRs, 3)
    # r["EFR"] = sma(EFRs, 10)
    r["ACSR"] = sma(CSRs, 10)
    r["AKG"] = sma(KGs, 3)
    r["AOEC"] = float(np.mean([r["AUKR"], r["AOKR"], r["ACSR"], r["AKG"]]))


    r["UKR(T)"] = UKRs[-1]
    r["OKR(T)"] = OKRs[-1]
    
    r["CSR(T)"] = CSRs[-1]
    r["KG(T)"] = KGs[-1]
    r["OEC(T)"] = float(np.mean([r["UKR(T)"], r["OKR(T)"],  r["CSR(T)"], r["KG(T)"]]))


    return r

    # def filter(OECTs):
    #     if not OECTs:
    #         return True
    #     if OECTs and f'{r["OEC(T)"]*100:.2f}' in OECTs:
    #         return True
    #     else:
    #         return False        
    # if filter(["45.77", "61.58", "65.09", "65.31", "66.62", "66.55", "67.40"]):
    #     print(f'{r["OEC(T)"]*100:.2f}', "#", path)
    #     return r
    # else:
    #     return None
    


os.chdir("experiments/results/nli/")
os.makedirs("csvs/", exist_ok=True)
result_files = []
for file in glob.glob("*.json"):
    result_files.append(file)

print(result_files)

results = []
for r_file in result_files:
    r = show_result(r_file)
    if r:
        results.append(r)
 
results.sort(key=lambda x:x["cl_method"])

results = pd.DataFrame(results)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

for ns_config in results.ns_config.unique():
    # print(ns_config)
    r = results[results["ns_config"]==ns_config]
    
    # r = r[((r["lr"]==3.5e-5) & (r["num_epochs"]==10)) | (r["cl_method"] == "none_cl") | (r["cl_method"] == "none_cl_offline_eval")]

    # r = r[(r["AEFR(T)"]>0.85) | (r["cl_method"]=="none_cl")]
    
    def _sort(column):
        # def tm_sorter(column):
        """Sort function"""
        cl_methods = ['none_cl', "simple_cl", "online_ewc", "er", "mir", "index_cl_bart_io_index"]
        correspondence = {team: order for order, team in enumerate(cl_methods)}
        return column.map(correspondence)
    r = r.sort_values(by=["steps", "lr", "num_epochs", "cl_method"])
    r = r.sort_values(by=["cl_method"], key = lambda x: x.str.len())
    r = r.sort_values(by="method_class", key=_sort, kind="mergesort")
    r = r.drop(columns=["ns_config", "method_class"])
    # r = r.drop(columns=["path"])
    r = r.drop(columns=["lr", "num_epochs"])
    r.to_csv(f"csvs/{ns_config}.csv", index=False, sep=",")
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

    