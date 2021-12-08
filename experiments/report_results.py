# %%

import json
import argparse
import pandas as pd
from argparse import Namespace
import numpy as np
import glob, os

from pandas.core import base
 
os.chdir("/private/home/yuchenlin/SemanticDebugger")
base_dir = "experiments/results/qa/" 
split = "test"
num_streams = 6

def sma(values):
    return float(np.mean(values)) 

def show_result(path): 
    # if path == "qa_nonecl_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_offline=yes_result.json":
    #     print()
    o = json.load(open(path))
    r = {}
    debugger_args = eval(o["debugger_args"])
    data_args = eval(o["data_args"])
    r["stream_id"] = data_args.stream_id
    path = path.replace(base_dir, "")
    r["path"] = path.replace(",", "|")

    r["noid_path"] = r["path"]

    for _ind in range(10):
        txt = f"[{_ind}]_result"
        if txt in r["noid_path"]:
            r["noid_path"] = r["noid_path"].replace(txt, "[]_result")

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
    if "-val" in ns_config_str:
        ns_config_str = ns_config_str.replace("-val", "")
        mode = "val"
    elif "-test" in ns_config_str:
        ns_config_str = ns_config_str.replace("-test", "")
        mode = "test"

    ns_config = eval(f"dict({ns_config_str})")
    r.update(ns_config)
     

    online = o["online_eval_results"]
    EFRs = [item["EFR"] for item in online]
    UKRs = [item["UKR"] for item in online if "UKR" in item]
    OKRs = [item["OKR"] for item in online if "OKR" in item]
    KGs = [item["KG"] for item in online if "KG" in item]
    CSRs = [item["CSR"] for item in online if "CSR" in item]
    if mode!="val" and len(EFRs) != ns_config["T"]:
        return None
    last_step = online[-1] 
    if last_step["timecode"] != ns_config["T"] -1:
        print(f"Error: ----> path={path}; the results doesn't match the length")
        return None
    
    r["AEFR(T)"] = float(np.mean(EFRs))

    r["AUKR"] = sma(UKRs)
    r["AOKR"] = sma(OKRs)
    r["ACSR"] = sma(CSRs)
    r["AKG"] = sma(KGs)
    r["AOEC"] = float(np.mean([r["AUKR"], r["AOKR"], r["ACSR"], r["AKG"]]))


    r["UKR(T)"] = UKRs[-1]
    r["OKR(T)"] = OKRs[-1]
    r["CSR(T)"] = CSRs[-1]
    r["KG(T)"] = KGs[-1]
    r["OEC(T)"] = float(np.mean([r["UKR(T)"], r["OKR(T)"],  r["CSR(T)"], r["KG(T)"]]))


    return r
# %%

def _sort(column):
    # def tm_sorter(column):
    """Sort function"""
    cl_methods = ['none_cl', "simple_cl", "online_ewc", "er", "mir", "index_cl_bart_io_index"]
    correspondence = {team: order for order, team in enumerate(cl_methods)}
    return column.map(correspondence)

# %%
if __name__ == '__main__':
    # %%
    
     
    os.makedirs(f"{base_dir}/csvs/", exist_ok=True)
    result_files = []

    

    for file in glob.glob(f'{base_dir}/*.json'):
        if split not in file:
            continue
        result_files.append(file)

    print(result_files)
    # %%

    results = []
    for r_file in result_files:
        # print(r_file)
        r = show_result(r_file)
        if r:
            results.append(r)
    
    # print(results)

    results.sort(key=lambda x:x["cl_method"])

    results = pd.DataFrame(results)

    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # %%
    results.to_csv(f"{base_dir}/csvs/full_results.csv", index=False, sep=",")

    for ns_config in results.ns_config.unique():
        # print(ns_config)
        r = results[results["ns_config"]==ns_config]
        
        # r = r[((r["lr"]==3.5e-5) & (r["num_epochs"]==10)) | (r["cl_method"] == "none_cl") | (r["cl_method"] == "none_cl_offline_eval")]
        items = []
        for noid_path in results.noid_path.unique():
            r_r = results[results["noid_path"]==noid_path]
            if len(r_r) != num_streams:
                print(f"{noid_path} does not have {num_streams} runs, so we skip it.")
                continue
            # %% 
            # print(r_r)
            records = r_r.to_dict("records")
            mean_item = records[0]
            keys = ["AEFR(T)", "AUKR", "AOKR", "ACSR", "AKG", "UKR(T)", "AOEC", "OKR(T)", "CSR(T)", "KG(T)", "OEC(T)"]
            for key in keys:
                mean_item[key] = r_r[key].mean()
            mean_item["OEC(T)-std"] = r_r["OEC(T)"].std()
            mean_item["stream_id"] = -1
            # print(mean_item)

            mean_item = [record for record in records if record["stream_id"]==5][0]  # TODO: debug only
            items.append(mean_item)
        r = pd.DataFrame(items)
        if "AEFR(T)" not in r:
            print()
        r = r[(r["AEFR(T)"]>=0.9) | (r["cl_method"]=="none_cl")]
        r = r.sort_values(by=["steps", "lr", "num_epochs", "cl_method"])
        r = r.sort_values(by=["cl_method"], key = lambda x: x.str.len())
        r = r.sort_values(by="method_class", key=_sort, kind="mergesort")
        r = r.drop(columns=["ns_config", "method_class", "path", "stream_id", "noid_path", "ACSR", "AOEC", "AKG", "AUKR", "AOKR"])
        # r = r.drop(columns=["lr", "num_epochs"])
        r.to_csv(f"{base_dir}/csvs/{ns_config}.csv", index=False, sep=",")
        print("-"*50)
        print(f'ns_config="{ns_config.replace(",", " & ")}",')
        print(open(f"{base_dir}/csvs/{ns_config}.csv").read())

        


     
# %%
