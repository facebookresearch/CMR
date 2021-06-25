import json
import numpy as np
import sys 
import os

from numpy.lib.function_base import median



# Load the json data
path = "logs/nq_dev_online_debug_result.json"
assert os.path.exists(path)
online_debug_result = json.load(open(path))

print(online_debug_result.keys())

def eval_forgetting(online_debug_result):
    pass_forgetting_data = []
    # Eval the forggeting issue.
    em_on_passes = []
    f1_on_passes = []
    for timecode, item in enumerate(online_debug_result["res_on_passes"]):
        result, result_all = item 
        # print(timecode, result["EM"])
        # d = dict(timecode=timecode, em=result["EM"], f1=result["QA-F1"])
        # pass_forgetting_data.append(d)
        em_on_passes.append(result["EM"])
        f1_on_passes.append(result["QA-F1"])
    worse = np.min(em_on_passes)
    mean = np.mean(em_on_passes)
    # median = np.median(em_on_passes)
    final = em_on_passes[-1]
    print(f"Forgetting measure (EM): worse={worse}; mean={mean}; final={final}")
    return worse, mean, final 

def eval_error_fixing(online_debug_result):
    final_state_bug_fixing_rate = online_debug_result["final_all_bug_eval"]["EM"]

    inter_prefix_efr = []
    inter_respon_efr = []

    bsz = 20
    odr = online_debug_result
    for timecode, ((before, after), em_fixed, f1_fixed, em_prefixed, f1_prefixed) in \
        enumerate(zip(odr["res_on_bugs"], odr["em_fixed_bugs"], odr["f1_fixed_bugs"], odr["em_prefixed_bugs"], odr["f1_prefixed_bugs"])):
        # em_before = before["EM"]
        # em_after = after["EM"]
        # f1_before = before["QA-F1"]
        # f1_after = after["QA-F1"] 
        # em_fix_rate = len(em_fixed)/(bsz-len(em_prefixed))
        # f1_fix_rate = len(f1_fixed)/(bsz-len(f1_prefixed))
        # em_improve = em_after - em_before
        # f1_improve = f1_after - f1_before
        inter_prefix_efr.append(len(em_prefixed)/bsz)
        inter_respon_efr.append(len(em_fixed)/(bsz-len(em_prefixed)))
    mean_ip_efr = np.mean(inter_prefix_efr)
    mean_ir_efr = np.mean(inter_respon_efr)
    print(f"Bug-Fixing measure (EM): final_state_bug_fixing_rate={final_state_bug_fixing_rate};")
    print(f"Bug-Fixing measure (EM): mean_ip_efr={mean_ip_efr}; mean_ir_efr={mean_ir_efr};")

eval_forgetting(online_debug_result)
eval_error_fixing(online_debug_result)


