import json

model0_instream_test = 0 
model0_replay_test = 0
def show_result(path, prefix):
    global model0_instream_test, model0_replay_test
    data = json.load(open(path))
    all_results = data["final_eval_results"]
    model0_instream_test = all_results["model0_instream_test"]["EM"]
    overall_perf = all_results["overall_oncoming_test"]["EM"]
    final_instream_test = all_results["final_instream_test"]["EM"]



    overall_error_number = all_results["overall_error_number"]
    overall_instant_fixing_rate = all_results["overall_instant_fixing_rate"]
    final_fixing_rate = all_results["final_fixing_rate"]["EM"] 


    final_upstream_test = all_results["final_upstream_test"]["EM"]


    final_upstream_test = all_results["final_upstream_test"]["EM"]

    overall_replay_test = all_results['overall_replay_test']["EM"] if 'overall_replay_test' in all_results else 0.0
    model0_replay_test = all_results['model0_replay_test']["EM"] if 'model0_replay_test' in all_results else 0.0

    f1 = 2*overall_replay_test*overall_perf/(overall_replay_test+overall_perf)

    train_steps = data["model_update_steps"]

    
    res = [prefix, f1, overall_replay_test, overall_perf, final_instream_test, overall_error_number, overall_instant_fixing_rate, final_fixing_rate, final_upstream_test, train_steps]
    print(",".join([str(r) for r in res]))

    
    

# print("prefix, overall_perf, final_instream_test, overall_error_number, overall_instant_fixing_rate, final_fixing_rate, final_upstream_test, train_steps")
print("method_name, f1, avg_replay_EM, avg_unseen_EM, final_retro_EM, overall_#error, avg_instant_efr, final_retro_efr, final_upstream_EM, #train_steps")

show_result("exp_results/dynamic_stream/cl_simple/nq_dev_0805_wr_dynamic_simplecl_result.json", "simple_cl")
show_result("exp_results/dynamic_stream/online_ewc/nq_dev_0805_wr_dynamic_ewc_result.json", "online_ewc")
show_result("exp_results/dynamic_stream/memory_based/nq_dev_0805_wr_er_result.json", "sparse_er")
show_result("exp_results/dynamic_stream/memory_based/nq_dev_0805_wr_mbpa_result.json", "mbpa")
show_result("exp_results/dynamic_stream/memory_based/nq_dev_0805_wr_mbpapp_result.json", "mbpa++")


prefix = "model_0"
overall_perf = model0_instream_test
final_instream_test = model0_instream_test
overall_error_number = (1-model0_instream_test)*32*100 # TODO:
overall_instant_fixing_rate = 0.0
final_fixing_rate = 0.0
final_upstream_test = 1.0
model0_replay_test = model0_replay_test
f1 = 2*model0_replay_test*overall_perf/(model0_replay_test+overall_perf)
train_steps = 0
res = [prefix, f1, model0_replay_test, overall_perf, final_instream_test, overall_error_number, overall_instant_fixing_rate, final_fixing_rate, final_upstream_test, train_steps]
print(",".join([str(r) for r in res]))