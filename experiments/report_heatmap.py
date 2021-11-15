maps = """MIR # 67.40 # QA_mir_lr=3e-5_ep=10_rs=32_rf=3_mcs=256_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
CFT # 61.58 # QA_simplecl_lr=3e-5_ep=10_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
ER # 66.62 # QA_er_lr=3e-5_ep=10_rs=32_rf=3_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
MaxLoss# 66.55 # QA_mir_lr=3e-5_ep=10_rs=32_rf=3_mcs=256_largest_afterloss_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
OnL2Reg # 65.09 # qa_simplecl_lr=3e-5_ep=10_l2w=1_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
OnEWC # 65.31 # qa_oewc_lr=3e-5_ep=10_lbd=250_gm=9e-1_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
*Frozen # 45.77 # QA_nonecl_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
"""
"""
*Frozen # 45.77 # QA_nonecl_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
"""
import os 
import json
import pandas as pd
import altair as alt

from semanticdebugger.notebooks.draw_utils import draw_stacked_bars, draw_curve

# os.chdir("experiments/results/qa/")
all_data = []
for line in maps.splitlines():
    name, OECT, path = [i.strip() for i in line.split("#")]
    # print(name, OECT, path)    
    o = json.load(open("experiments/results/qa_backup_1113/" + path))
    # debugger_args = eval(o["debugger_args"])
    # data_args = eval(o["data_args"])
    r = o["online_eval_results"]
    for item in r:
        item["prefix"] = name
        if item["timecode"] == 99: 
            item["timecode"] += 1
    all_data += r
    # print(o)
    # EFRs = [item["EFR"] for item in online]
    # UKRs = [item["UKR"] for item in online if "UKR" in item]
    # OKRs = [item["OKR"] for item in online if "OKR" in item]
    # KGs = [item["KG"] for item in online if "KG" in item]
    # CSRs = [item["CSR"] for item in online if "CSR" in item]

# for item in all_data:
#     if item["name"] == "*Frozne":
#         item["OKR"] = 0
#     else:
#         if item["OKR"]

all_data = pd.DataFrame(all_data)
# all_data = all_data.drop(columns=["before_eval_results", "before_error_ids", "mir_buffer_ids", "retrieved_ids", "OKR_sampled_ids"])

def flatten_list(lst):
    lst = list(lst)
    flst = []
    for l in lst:
        flst += l
    return flst

def flatten_predictions(before_eval_results):
    lst = list(before_eval_results)
    flst = []
    scores = []
    for l in lst:
        flst += l["predictions"]
        # scores += l["metric_results"]["EM"] 
    return flst, scores

def jaccard(list1, list2):
    list1 = set(list1)
    list2 = set(list2)
    intersection = len(list1 & list2)
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def error_sim(all_data, span=[0, 10], methods=["CFT", "OnL2Reg", "OnEWC",  "ER", "MaxLoss", "MIR"]):
    df = all_data[(all_data.timecode<=span[1]) & (all_data.timecode>=span[0])] 
    sims = []
    for method_1 in methods:
        for method_2 in methods:
            if method_1 == method_2:
                continue
            # errors1 = flatten_list(df[df.prefix==method_1]["before_error_ids"])
            # errors2 = flatten_list(df[df.prefix==method_2]["before_error_ids"])
            errors1, scores1 = flatten_predictions(df[df.prefix==method_1]["before_eval_results"])
            errors2, scores2 = flatten_predictions(df[df.prefix==method_2]["before_eval_results"])
            # if len(errors1) == 0:
            #     continue
            assert len(errors1) == len(errors2)
            sim = sum([p1!=p2 for p1, p2 in zip(errors1, errors2)])/len(errors1)
            # sim = jaccard(errors1, errors2)
            sims.append({"method1": method_1, "method2": method_2, "sim": sim})
            print(f"{method_1}-{method_2}: {sim}")
    sims = pd.DataFrame(sims)
    fig = alt.Chart(sims).mark_rect().encode(
        x=alt.X('method1:O', sort=methods),
        y=alt.Y('method2:O', sort=methods),
        # color='sim:Q'
        color = alt.Color('sim:Q',scale=alt.Scale(domain=[0.35, 0.45]))
    )
    fig = fig.properties(width=500, height=500).configure_title(fontSize=0,
    ).configure_axis(
        labelFontSize=30,
        titleFontSize=0,  
    )
    fig.save(f"figures/heatmaps/{span[0]}-{span[1]}.png", scale=5.0)

error_sim(all_data, span=[0,10])
error_sim(all_data, span=[10,20])
error_sim(all_data, span=[20,30])
error_sim(all_data, span=[30,40])
error_sim(all_data, span=[50,60])
error_sim(all_data, span=[90,100])

error_sim(all_data, span=[0,20])
error_sim(all_data, span=[40,60])
error_sim(all_data, span=[80,100])
# all_data = all_data.dropna()
# all_data['OEC'] = all_data.drop(columns=["timecode", "EFR", "SR", "Overall"]).mean(numeric_only=True, axis=1)
# print(all_data)
# exit()
# print(all_data)

# fig = draw_curve(df=all_data[all_data["EFR"].notnull()], fig_title=f"EFR", y_scale=[0.7, 1], x_key="timecode:O", y_key="EFR:Q", y_title="EFR")
# fig.save('figures/curves/EFRs.png', scale_factor=2.0)


# color_dom = ["CFT", "OnL2Reg", "OnEWC",  "ER", "MaxLoss", "MIR"]
# color_range = ['gray', 'brown',  '#7fc97f', '#D35400', 'purple', '#386cb0']

# fig = draw_curve(df=all_data[all_data["UKR"].notnull()], fig_title=f"UKR", y_scale=[0.66, 0.82],  x_key="timecode:O", y_key="UKR:Q", y_title="UKR", height=600, width=500, orient="none", color_dom=color_dom, color_range=color_range)
# fig.save('figures/curves/UKRs.png', scale_factor=3.0)

# fig = draw_curve(df=all_data[all_data["OKR"].notnull()], fig_title=f"OKR", y_scale=[0.77, 0.96],  x_key="timecode:O", y_key="OKR:Q", y_title="OKR", height=600, width=500, orient="none", color_dom=color_dom, color_range=color_range)
# fig.save('figures/curves/OKRs.png', scale_factor=3.0)

# fig = draw_curve(df=all_data[all_data["CSR"].notnull()], fig_title=f"CSR", y_scale=[0.52, 0.67],  x_key="timecode:O", y_key="CSR:Q", y_title="CSR", height=600, width=500, orient="none", color_dom=color_dom, color_range=color_range)
# fig.save('figures/curves/CSRs.png', scale_factor=3.0)

# fig = draw_curve(df=all_data[all_data["KG"].notnull()], fig_title=f"KG", y_scale=[0.43, 0.54],  x_key="timecode:O", y_key="KG:Q", y_title="KG", height=600, width=500, orient="none", color_dom=color_dom, color_range=color_range)
# fig.save('figures/curves/KGs.png', scale_factor=3.0)

# fig = draw_curve(df=all_data[all_data["OEC"].notnull()], fig_title=f"OEC", y_scale=[0.61, 0.72],  x_key="timecode:O", y_key="OEC:Q", y_title="OEC", height=600, width=500, orient="none", color_dom=color_dom, color_range=color_range)
# fig.save('figures/curves/OECs.png', scale_factor=3.0)