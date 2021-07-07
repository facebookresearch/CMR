import json
from altair.vegalite.v4.api import value
import numpy as np
import sys 
import os

from numpy.lib.function_base import median




def eval_forgetting(online_debug_result, timecodes):
    pass_forgetting_data = []
    # Eval the forggeting issue.
    em_on_passes = []
    f1_on_passes = []
    for timecode in timecodes:
        item = online_debug_result[str(timecode)]
        r = item["eval_results_overall_forget"]["metric_results"]
        em_on_passes.append(r["EM"])
        f1_on_passes.append(r["QA-F1"])
    worse = np.min(em_on_passes)
    mean = np.mean(em_on_passes)
    # median = np.median(em_on_passes)
    final = em_on_passes[-1]
    # print(f"Forgetting measure (EM): worse={worse}; mean={mean}; final={final}")
    return worse, mean, final 

def eval_error_fixing(online_debug_result, timecodes):
    final_state_bug_fixing_rate = online_debug_result[str(timecodes[-1])]["eval_results_overall_bug"]["metric_results"]["EM"]

    inter_prefix_efr = []
    inter_respon_efr = []

    bsz = 20
    odr = online_debug_result
    # TODO: add these back later.
    # for timecode, ((before, after), em_fixed, f1_fixed, em_prefixed, f1_prefixed) in \
    #     enumerate(zip(odr["res_on_bugs"], odr["em_fixed_bugs"], odr["f1_fixed_bugs"], odr["em_prefixed_bugs"], odr["f1_prefixed_bugs"])):
        
    #     inter_prefix_efr.append(len(em_prefixed)/bsz)
    #     inter_respon_efr.append(len(em_fixed)/(bsz-len(em_prefixed)))
    # mean_ip_efr = np.mean(inter_prefix_efr)
    # mean_ir_efr = np.mean(inter_respon_efr)
    # print(f"Bug-Fixing measure (EM): final_state_bug_fixing_rate={final_state_bug_fixing_rate};")
    # print(f"Bug-Fixing measure (EM): mean_ip_efr={mean_ip_efr}; mean_ir_efr={mean_ir_efr};")
    mean_ip_efr, mean_ir_efr = 0, 0
    return final_state_bug_fixing_rate, mean_ip_efr, mean_ir_efr

def print_eval(path="bug_data/output/nq_dev_0625_1e-5_e3_result.json"):
    # Load the json data
    lr = path.split("_")[-3]
    num_epoch = path.split("_")[-2][1:]
    assert os.path.exists(path)
    all_results = json.load(open(path))
    # print(output_info.keys()) 
    # online_debug_results = output_info["online_debug_results"]
    timecodes = [int(t) for t in list(all_results.keys())]
    timecodes = sorted(timecodes, reverse=False)
    worse_kr, mean_kr, final_kr = eval_forgetting(all_results, timecodes)
    final_efr, mean_ip_efr, mean_ir_efr = eval_error_fixing(all_results, timecodes)
    print(f"{lr}, {num_epoch}, {worse_kr}, {mean_kr}, {final_kr}, {mean_ip_efr}, {mean_ir_efr}, {final_efr}")

def aggregate_offline_results(path="bug_data/output/nq_dev_0701_v2_offline_eval/"):
    import glob
    alltime_results = {}
    for thread_res_path in sorted(glob.glob(f"{path}/thread_*.json")):
        with open(thread_res_path) as f:
            thread_res = json.load(f)
        # for key, values in single_res.items():
        #     if key not in alltime_results:
        #         alltime_results[key] = []
            # alltime_results[key] += values
        alltime_results.update(thread_res)

    with open(f"{path}/alltime_result.json", "w") as f:
        json.dump(alltime_results, f)
    


if __name__ == '__main__':    
    aggregate_offline_results("bug_data/output/nq_dev_0706_3e-5_e5_offline_eval")
    # aggregate_offline_results("bug_data/output/nq_dev_0706_3e-5_e3_offline_eval")
    # aggregate_offline_results("bug_data/output/nq_dev_0706_1e-5_e3_offline_eval")
    # aggregate_offline_results("bug_data/output/nq_dev_0706_1e-5_e5_offline_eval")
    
    print("{lr}, {num_epoch}, {worse_kr}, {mean_kr}, {final_kr}, {mean_ip_efr}, {mean_ir_efr}, {final_efr}")
    print_eval("bug_data/output/nq_dev_0706_3e-5_e5_offline_eval/alltime_result.json")

    # print_eval("bug_data/output/nq_dev_0625_1e-5_e3_result.json")
    # print_eval("bug_data/output/nq_dev_0625_3e-5_e3_result.json")
    # print_eval("bug_data/output/nq_dev_0625_1e-5_e5_result.json")
    # print_eval("bug_data/output/nq_dev_0625_3e-5_e5_result.json")
    
    # print_eval("bug_data/output/nq_dev_0701v3_1e-5_e3_result.json")
    # print_eval("bug_data/output/nq_dev_0701v3_1e-5_e5_result.json")
    # print_eval("bug_data/output/nq_dev_0701v3_3e-5_e3_result.json")
    # print_eval("bug_data/output/nq_dev_0701v3_3e-5_e5_result.json")