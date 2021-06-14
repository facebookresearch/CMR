import json 

result_path = "logs/online_debug_result.json"
with open(result_path) as f:
    results = json.load(f)
    results_on_bugs = results["results_on_bugs"]
    results_on_passes = results["results_on_passes"]

# for timecode, res in enumerate(results_on_passes):
#     print(timecode, res[0]["EM"], res[0]["QA-F1"])

for timecode, res in enumerate(results_on_bugs):
    print(timecode, res[0]["EM"], res[1]["EM"], res[0]["QA-F1"], res[1]["QA-F1"])