import json
import numpy as np
 

upstream_ex_ids = set()

def show_memory_ratio(path, prefix, ): 
    if len(upstream_ex_ids) == 0:
        with open("exp_results/data_streams/mrqa.nq_train.memory.jsonl") as f:
            lines = f.read().splitlines()
            for line in lines:
                item = json.loads(line)
                upstream_ex_ids.add(item["id"]) 

    data = json.load(open(path)) 
    ratios = []
    for time_item in  data["online_eval_results"]:
        if "retrieved_ids" in time_item:
            retrieved_ids = set(time_item["retrieved_ids"])
            upstream_ratio = len(retrieved_ids.intersection(upstream_ex_ids)) / len(retrieved_ids)
            ratios.append({"timecode": time_item["timecode"], "upstream_ratio": upstream_ratio, "online_ratio": 1-upstream_ratio})
    print(ratios)
     
    



def show_result(path, prefix, return_data=False): 
    data = json.load(open(path))
    all_results = data["final_eval_results"] 
    overall_perf = all_results["overall_oncoming_test"]["EM"]
    final_instream_test = all_results["final_instream_test"]["EM"]

    overall_perf_f1 = all_results["overall_oncoming_test"]["QA-F1"]
    final_instream_test_f1 = all_results["final_instream_test"]["QA-F1"]



    overall_error_number = all_results["overall_error_number"]
    overall_instant_fixing_rate = all_results["overall_instant_fixing_rate"] 
    final_fixing_rate = -1


    final_upstream_test = all_results["final_upstream_test"]["EM"]
    final_upstream_test_f1 = all_results["final_upstream_test"]["QA-F1"]

    overall_replay_test = all_results['overall_replay_test']["EM"] if 'overall_replay_test' in all_results else 0.0 

    train_steps = data["model_update_steps"]

     
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
    # print(f"{prefix}-std, "+",".join([str(r) for r in std]))
    # print(f"{prefix}-min, "+",".join([str(r) for r in min]))
    print(f"{prefix}-max, "+",".join([str(r) for r in max]))
    print()

    
    

 
print("method_name, avg_unseen_EM, avg_unseen_F1, final_retro_EM, final_retro_F1, overall_#error, avg_instant_efr, final_retro_efr, final_upstream_EM, final_upstream_F1, #train_steps")

 
show_result("exp_results/dynamic_stream/none/results/nq_dev_0922_none_mixed_allerrors_result.json", "None") 


# print("T=100 and rq=3")

# show_result("exp_results/dynamic_stream/none/results/0924_MixedAllError_T=100_nonecl_seed=42_result.json", "None")




# show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllError_T=100_simplecl_seed=$seed_result.json", "simple_cl", random_seeds=["42", "0212", "1213"]) 

 

# show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=random_seed=$seed_result.json", "MIR_c256_random", random_seeds=["42", "0212", "1213"]) 

# print("MixedAllError_MIR_rs32_rf1")
  
# show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_er_M=U+I_rs=32_rq=3_seed=$seed_result.json", "ER", random_seeds=["42", "0212", "1213"]) 

# show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=none_seed=$seed_result.json", "MIR_c256", random_seeds=["42", "0212", "1213"]) 

# show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=reverse_seed=$seed_result.json", "MIR_c256_reverse", random_seeds=["42", "0212", "1213"]) 
   
# show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=largest_afterloss_seed=$seed_result.json", "MIR_c256_largest_afterloss", random_seeds=["42", "0212", "1213"]) 

# show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=512_mode=none_seed=$seed_result.json", "MIR_c512", random_seeds=["42", "0212", "1213"]) 

# show_result_stat("exp_results/dynamic_stream/memory_based/results/0924_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=1024_mode=none_seed=$seed_result.json", "MIR_c1024", random_seeds=["42", "0212", "1213"]) 

 

 


print("09/25/2021 8 rounds stat")



show_result_stat("exp_results/dynamic_stream/cl_simple/results/0926_MixedAllError_T=100_simple_ep=5_seed=$seed_result.json", "SimpleCL ep=5", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/cl_simple/results/0926_MixedAllError_T=100_simple_ep=7_seed=$seed_result.json", "SimpleCL ep=7", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/cl_simple/results/0926_MixedAllError_T=100_simple_ep=8_seed=$seed_result.json", "SimpleCL ep=8", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 
show_result_stat("exp_results/dynamic_stream/cl_simple/results/0926_MixedAllError_T=100_simple_ep=10_seed=$seed_result.json", "SimpleCL ep=10", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

# show_result_stat("exp_results/dynamic_stream/memory_based/results/0925_MixedAllErrors_T=100_er_M=U+I_rs=32_rq=3_seed=$seed_result.json", "ER", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0927_MixedAllErrors_T=100_er_M=U+I_rs=32_rq=3_seed=$seed_result.json", "ER", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

# show_result_stat("exp_results/dynamic_stream/memory_based/results/0927_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=32_mode=random_seed=$seed_result.json", "MIR_c32-random", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0925_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=none_seed=$seed_result.json", "MIR_c256", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0925_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=1024_mode=none_seed=$seed_result.json", "MIR_c1024", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0925_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=4096_mode=none_seed=$seed_result.json", "MIR_c4096", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0925_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=reverse_seed=$seed_result.json", "MIR_c256-reverse", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0925_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=largestloss_seed=$seed_result.json", "MIR_c256-largestloss", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 


print("*"*100)

show_result_stat("exp_results/dynamic_stream/index_based/results/0929_MixedAllErrors_T=100_index_M=U+I_rs=32_rq=3_seed=$seed_rank=most_different_result.json", "Index-different", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/index_based/results/0929_MixedAllErrors_T=100_index_M=U+I_rs=32_rq=3_seed=$seed_rank=most_similar_result.json", "Index-similar", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 


print("*"*100)




show_result_stat("exp_results/dynamic_stream/memory_based/results/0930_MixedAllErrors_T=100_er_M=U+I_rs=32_rq=1_seed=$seed_result.json", "ER_rf1", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0930_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=1_candidate=256_mode=none_seed=$seed_result.json", "MIR_c256_rf1", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/memory_based/results/0930_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=none_seed=$seed_result.json", "MIR_c256_rf3", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 


print("*"*100)

show_result_stat("exp_results/dynamic_stream/index_based/results/0930_MixedAllErrors_T=100_index_M=U+I_rs=32_rq=3_seed=$seed_rank=most_similar_result.json", "Index-similar", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 




# show_result_stat("exp_results/dynamic_stream/index_based/results/0930_MixedAllErrors_T=100_index_M=U+I_rs=32_rq=3_rank=most_similar_mir=no(256)_seed=$seed_result.json", "Index-similar-mir=no", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

# show_result_stat("exp_results/dynamic_stream/index_based/results/0930_MixedAllErrors_T=100_index_M=U+I_rs=32_rq=3_rank=most_similar_mir=yes(256)_seed=$seed_result.json", "Index-similar-mir=yes", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"])


show_result_stat("exp_results/dynamic_stream/index_based/results/1001_MixedAllErrors_T=100_index_M=U+I_rs=32_rq=3_rank=most_similar_mir=no(0)_seed=$seed_result.json", "Index-similar-mir=no-top_each", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

# show_result_stat("exp_results/dynamic_stream/index_based/results/1001v2_MixedAllErrors_T=100_index_M=U+I_rs=32_rq=3_rank=most_different_mir=no(0)_seed=$seed_result.json", "Index-different-mir=no-top_each", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 


# show_result_stat("exp_results/dynamic_stream/index_based/results/1001v2_MixedAllErrors_T=100_index_M=U+I_rs=32_rq=3_rank=most_similar_mir=yes(256)_seed=$seed_rdi'vhvlcufngfcehbkrddkeefekhlbkhcesult.json", "Index-similar-mir=yes-top_each", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"])
# 
# 

print("-"*20+"1012"+"-"*20)

show_result_stat("exp_results/dynamic_stream/index_based/results/1012_MixedAllErrors_T=100_biencoder_M=U+I_rs=32_rq=3_rank=most_similar_mir=no(0)_seed=$seed_result.json", "Biencoder-similar-mir=no-top_each", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

print("-"*20+"1014"+"-"*20)


show_result_stat("exp_results/dynamic_stream/memory_based/results/1014_MixedAllErrors_T=50_er_M=U+I_rs=32_rq=1_seed=$seed_result.json", "ER-T50-F1", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 


show_result_stat("exp_results/dynamic_stream/memory_based/results/1014_MixedAllErrors_T=50_mir_M=U+I_rs=32_rq=1_candidate=256_mode=none_seed=$seed_result.json", "MIR-C256-T50-F1", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 

show_result_stat("exp_results/dynamic_stream/index_based/results/1014_MixedAllErrors_T=50_biencoder_M=U+I_rs=32_rq=1_rank=most_similar_mir=no(0)_seed=$seed_result.json", "BIENCODER-T50-F1", random_seeds=["42", "0212", "1213", "2021", "123", "456", "567", "789"]) 



show_result_stat("exp_results/dynamic_stream/memory_based/results/1014_MixedAllErrors_T=100_er_M=U+I_rs=32_rq=3_seed=$seed_result.json","ER-T100-F3", random_seeds=["42", "0212", "2021", "123", "456", "567", "789"])

show_result_stat("exp_results/dynamic_stream/memory_based/results/1014_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=none_seed=$seed_result.json","MIR-C256-T100-F3", random_seeds=["42", "0212", "2021", "123", "456", "567", "789"])  #  "1213",

show_result_stat("exp_results/dynamic_stream/index_based/results/1014v4_MixedAllErrors_T=100_biencoder_M=U+I_rs=32_rq=3_rank=most_similar_mir=no(0)_seed=$seed_result.json","BIENCODER-T100-F3", random_seeds=["42", "0212", "2021", "123", "456", "567", "789"])  #  "1213",



show_result_stat("exp_results/dynamic_stream/index_based/results/1001v2_MixedAllErrors_T=100_index_M=U+I_rs=32_rq=3_rank=most_similar_mir=no(0)_seed=$seed_result.json","BartIndex-T100-F3", random_seeds=["42", "0212", "2021", "123", "456", "567", "1213", "789"])  # 

show_result_stat("exp_results/dynamic_stream/index_based/results/1019_MixedAllErrors_T=100_index_M=U+I_rs=32_rq=3_rank=most_similar_mir=no(0)_seed=$seed_result.json","BartIndex-T100-F3", random_seeds=["42", "0212", "2021", "123", "456", "567", "1213", "789"])  # 

show_result_stat("exp_results/dynamic_stream/index_based/results/1019_MixedAllErrors_T=100_IOindex_M=U+I_rs=32_rq=3_rank=most_similar_mir=no(0)_seed=$seed_result.json","BartIOIndex-T100-F3", random_seeds=["42", "0212", "2021", "123", "456", "567",  "1213","789"])  #  "1213",




# show_memory_ratio("exp_results/dynamic_stream/memory_based/results/1014_MixedAllErrors_T=100_er_M=U+I_rs=32_rq=3_seed=42_result.json", "ER-T100-F3",)
# show_memory_ratio("exp_results/dynamic_stream/memory_based/results/1014_MixedAllErrors_T=100_mir_M=U+I_rs=32_rq=3_candidate=256_mode=none_seed=42_result.json", "MIR-C256-T100-F3",)