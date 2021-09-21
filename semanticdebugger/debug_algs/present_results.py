import json

model0_instream_test = 0 
model0_replay_test = 0
def show_result(path, prefix):
    # global model0_instream_test, model0_replay_test
    data = json.load(open(path))
    all_results = data["final_eval_results"]
    # model0_instream_test = all_results["model0_instream_test"]["EM"]
    overall_perf = all_results["overall_oncoming_test"]["EM"]
    final_instream_test = all_results["final_instream_test"]["EM"]



    overall_error_number = all_results["overall_error_number"]
    overall_instant_fixing_rate = all_results["overall_instant_fixing_rate"]
    final_fixing_rate = all_results["final_fixing_rate"]["EM"] 


    final_upstream_test = all_results["final_upstream_test"]["EM"]


    final_upstream_test = all_results["final_upstream_test"]["EM"]

    overall_replay_test = all_results['overall_replay_test']["EM"] if 'overall_replay_test' in all_results else 0.0
    # model0_replay_test = all_results['model0_replay_test']["EM"] if 'model0_replay_test' in all_results else 0.0

    f1 = 2*overall_replay_test*overall_perf/(overall_replay_test+overall_perf)

    train_steps = data["model_update_steps"]

    
    # res = [prefix, f1, overall_replay_test, overall_perf, final_instream_test, overall_error_number, overall_instant_fixing_rate, final_fixing_rate, final_upstream_test, train_steps]
    res = [prefix, overall_perf, final_instream_test, overall_error_number, overall_instant_fixing_rate, final_fixing_rate, final_upstream_test, train_steps]
    print(",".join([str(r) for r in res]))

    
    

# print("prefix, overall_perf, final_instream_test, overall_error_number, overall_instant_fixing_rate, final_fixing_rate, final_upstream_test, train_steps")
# print("method_name, f1, avg_replay_EM, avg_unseen_EM, final_retro_EM, overall_#error, avg_instant_efr, final_retro_efr, final_upstream_EM, #train_steps")
print("method_name, avg_unseen_EM, final_retro_EM, overall_#error, avg_instant_efr, final_retro_efr, final_upstream_EM, #train_steps")


# show_result("exp_results/dynamic_stream/none/results/nq_dev_0813_wr_wpara_dynamic_none_result.json", "none_cl")
# show_result("exp_results/dynamic_stream/cl_simple/results/nq_dev_0813_wr_wpara_dynamic_simplecl_result.json", "simple_cl")
# show_result("exp_results/dynamic_stream/online_ewc/results/nq_dev_0813_wr_wpara_dynamic_ewc_result.json", "online_ewc")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0813_wr_wpara_er_result.json", "sparse_er")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0813_wr_wpara_mbpa_result.json", "mbpa")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0813_wr_wpara_mbpapp_result.json", "mbpa++")

# print()

# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0916_wr_wpara_er_seed=42_result.json", "er_seed_1")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0916_wr_wpara_er_seed=0212_result.json", "er_seed_2")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0916_wr_wpara_er_seed=1213_result.json", "er_seed_3")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0916_wr_wpara_er_seed=2021_result.json", "er_seed_4")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0916_wr_wpara_er_seed=666_result.json", "er_seed_5")

# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0916_wr_wpara_mir_seed=42_result.json", "mir_seed_1")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0916_wr_wpara_mir_seed=0212_result.json", "mir_seed_2")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0916_wr_wpara_mir_seed=1213_result.json", "mir_seed_3")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0916_wr_wpara_mir_seed=2021_result.json", "mir_seed_4")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0916_wr_wpara_mir_seed=666_result.json", "mir_seed_5")

# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_mir_candidate=1024_seed=42_result.json", "mir_1024_seed_1")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_mir_candidate=1024_seed=0212_result.json", "mir_1024_seed_2")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_mir_candidate=1024_seed=2021_result.json", "mir_1024_seed_3")




# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_er_freq=3_seed=42_result.json", "er_seed_1")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_er_freq=3_seed=2021_result.json", "er_seed_1")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_er_freq=3_seed=0212_result.json", "er_seed_1")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_er_freq=3_seed=1213_result.json", "er_seed_1")

# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_mir_freq=3_candidate=2048_seed=42_result.json", "mir_2048_seed_1")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_mir_freq=3_candidate=2048_seed=2021_result.json", "mir_2048_seed_2")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_mir_freq=3_candidate=2048_seed=0212_result.json", "mir_2048_seed_3")
# show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0917_wr_wpara_mir_freq=3_candidate=2048_seed=1213_result.json", "mir_2048_seed_4")


show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_er_mix=Yes_freq=3_seed=42_result.json", "er_mix=Yes_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_er_mix=Yes_freq=3_seed=2021_result.json", "er_mix=Yes_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_er_mix=Yes_freq=3_seed=0212_result.json", "er_mix=Yes_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_er_mix=Yes_freq=3_seed=1213_result.json", "er_mix=Yes_seed_1")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_freq=3_candidate=256=42_result.json", "mir_mix=Yes_256_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_freq=3_candidate=256=2021_result.json", "mir_mix=Yes_256_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_freq=3_candidate=256=0212_result.json", "mir_mix=Yes_256_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_freq=3_candidate=256=1213_result.json", "mir_mix=Yes_256_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_mix=Yes_freq=3_candidate=512_seed=42_result.json", "mir_mix=Yes_512_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_mix=Yes_freq=3_candidate=512_seed=2021_result.json", "mir_mix=Yes_512_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_mix=Yes_freq=3_candidate=512_seed=0212_result.json", "mir_mix=Yes_512_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_mix=Yes_freq=3_candidate=512_seed=1213_result.json", "mir_mix=Yes_512_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_reverse=Yes_mix=Yes_freq=3_candidate=256_seed=42_result.json", "mir_reverse=Yes_mix=Yes_256_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_reverse=Yes_mix=Yes_freq=3_candidate=256_seed=2021_result.json", "mir_reverse=Yes_mix=Yes_256_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_reverse=Yes_mix=Yes_freq=3_candidate=256_seed=0212_result.json", "mir_reverse=Yes_mix=Yes_256_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_reverse=Yes_mix=Yes_freq=3_candidate=256_seed=1213_result.json", "mir_reverse=Yes_mix=Yes_256_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_absolute=Yes_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=42_result.json", "mir_absolute=Yes_meanloss=Yes_mix=Yes_256_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_absolute=Yes_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=2021_result.json", "mir_absolute=Yes_meanloss=Yes_mix=Yes_256_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_absolute=Yes_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=0212_result.json", "mir_absolute=Yes_meanloss=Yes_mix=Yes_256_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0919_wr_wpara_mir_absolute=Yes_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=1213_result.json", "mir_absolute=Yes_meanloss=Yes_mix=Yes_256_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=64_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_64_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=64_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_64_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=64_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_64_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=64_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_64_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=128_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_128_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=128_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_128_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=128_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_128_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=128_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_128_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=512_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_512_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=512_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_512_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=512_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_512_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=512_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_512_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=1024_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_1024_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=1024_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_1024_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=1024_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_1024_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=1024_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_1024_seed_4")



print()
print()


show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=16_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_16_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=16_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_16_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=16_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_16_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=16_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_16_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=64_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_64_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=64_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_64_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=64_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_64_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=64_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_64_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=256_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_256_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=512_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_512_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=512_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_512_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=512_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_512_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=512_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_512_seed_4")
print()
print()
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=1024_seed=42_result.json", "mir_meanloss=Yes_mix=Yes_1024_seed_1")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=1024_seed=2021_result.json", "mir_meanloss=Yes_mix=Yes_1024_seed_2")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=1024_seed=0212_result.json", "mir_meanloss=Yes_mix=Yes_1024_seed_3")
show_result("exp_results/dynamic_stream/memory_based/results/nq_dev_0920v2_wr_wpara_mir_meanloss=Yes_mix=Yes_freq=3_candidate=1024_seed=1213_result.json", "mir_meanloss=Yes_mix=Yes_1024_seed_4")



# prefix = "model_0"
# overall_perf = model0_instream_test
# final_instream_test = model0_instream_test
# overall_error_number = (1-model0_instream_test)*32*100 # TODO:
# overall_instant_fixing_rate = 0.0
# final_fixing_rate = 0.0
# final_upstream_test = 1.0
# model0_replay_test = model0_replay_test
# f1 = 2*model0_replay_test*overall_perf/(model0_replay_test+overall_perf)
# train_steps = 0
# res = [prefix, f1, model0_replay_test, overall_perf, final_instream_test, overall_error_number, overall_instant_fixing_rate, final_fixing_rate, final_upstream_test, train_steps]
# print(",".join([str(r) for r in res]))