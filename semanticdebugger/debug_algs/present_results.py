import json
import numpy as np

model0_instream_test = 0 
model0_replay_test = 0
def show_result(path, prefix, return_data=False):
    # global model0_instream_test, model0_replay_test
    data = json.load(open(path))
    all_results = data["final_eval_results"]
    # model0_instream_test = all_results["model0_instream_test"]["EM"]
    overall_perf = all_results["overall_oncoming_test"]["EM"]
    final_instream_test = all_results["final_instream_test"]["EM"]

    overall_perf_f1 = all_results["overall_oncoming_test"]["QA-F1"]
    final_instream_test_f1 = all_results["final_instream_test"]["QA-F1"]



    overall_error_number = all_results["overall_error_number"]
    overall_instant_fixing_rate = all_results["overall_instant_fixing_rate"]
    # final_fixing_rate = all_results["final_fixing_rate"]["EM"] 
    final_fixing_rate = -1


    final_upstream_test = all_results["final_upstream_test"]["EM"]
    final_upstream_test_f1 = all_results["final_upstream_test"]["QA-F1"]

    overall_replay_test = all_results['overall_replay_test']["EM"] if 'overall_replay_test' in all_results else 0.0
    # model0_replay_test = all_results['model0_replay_test']["EM"] if 'model0_replay_test' in all_results else 0.0

    # f1 = 2*overall_replay_test*overall_perf/(overall_replay_test+overall_perf)

    train_steps = data["model_update_steps"]

    
    # res = [prefix, f1, overall_replay_test, overall_perf, final_instream_test, overall_error_number, overall_instant_fixing_rate, final_fixing_rate, final_upstream_test, train_steps]
    res = [prefix, overall_perf, overall_perf_f1, final_instream_test, final_instream_test_f1, overall_error_number, overall_instant_fixing_rate, final_fixing_rate, final_upstream_test, final_upstream_test_f1, train_steps]
    if return_data:
        return res
    else:
        print(",".join([str(r) for r in res]))

    



def show_result_stat(path, prefix, random_seeds):
    results = []
    # names = "prefix, overall_perf, overall_perf_f1, final_instream_test, final_instream_test_f1, overall_error_number, overall_instant_fixing_rate, final_fixing_rate, final_upstream_test, final_upstream_test_f1, train_steps"

    for seed in random_seeds:
        _path = path.replace("$seed", seed) 
        res = show_result(_path, prefix, return_data=True)
        item = {}
        # for key, value in zip(names.split(","), res):
        #     item[key.strip()] = value 
        results.append(res[1:])
    # print(results)
    # mean 
    results = np.array(results)
    mean = np.mean(results, axis=0)
    min = np.min(results, axis=0)
    max = np.max(results, axis=0)
    std = np.std(results, axis=0)
    print()
    print(f"{prefix}-mean, "+",".join([str(r) for r in mean]))
    print(f"{prefix}-max, "+",".join([str(r) for r in max]))
    print()
    # print(f"{prefix}-min, "+",".join([str(r) for r in min]))
    # print(f"{prefix}-std, "+",".join([str(r) for r in std]))

    
    

 
print("method_name, avg_unseen_EM, avg_unseen_F1, final_retro_EM, final_retro_F1, overall_#error, avg_instant_efr, final_retro_efr, final_upstream_EM, final_upstream_F1, #train_steps")

 
show_result("exp_results/dynamic_stream/none/results/nq_dev_0922_none_mixed_allerrors_result.json", "None") 


print("T=100 and rq=3")

show_result("exp_results/dynamic_stream/none/results/0924_MixedAllError_T=100_nonecl_seed=42_result.json", "None")




show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllError_T=100_simplecl_seed=$seed_result.json", "simple_cl", random_seeds=["42", "0212", "1213"]) 

 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=random_seed=$seed_result.json", "MIR_c256_random", random_seeds=["42", "0212", "1213"]) 

print("MixedAllError_MIR_rs32_rf1")
  
show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_er_M=U+I_rs=32_rq=3_seed=$seed_result.json", "ER", random_seeds=["42", "0212", "1213"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=none_seed=$seed_result.json", "MIR_c256", random_seeds=["42", "0212", "1213"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=reverse_seed=$seed_result.json", "MIR_c256_reverse", random_seeds=["42", "0212", "1213"]) 
   
show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=largest_afterloss_seed=$seed_result.json", "MIR_c256_largest_afterloss", random_seeds=["42", "0212", "1213"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=512_mode=none_seed=$seed_result.json", "MIR_c512", random_seeds=["42", "0212", "1213"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=1024_mode=none_seed=$seed_result.json", "MIR_c1024", random_seeds=["42", "0212", "1213"]) 

 



  