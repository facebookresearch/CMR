# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

maps = """MIR # 67.40 # QA_mir_lr=3e-5_ep=10_rs=32_rf=3_mcs=256_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
CFT # 61.58 # QA_simplecl_lr=3e-5_ep=10_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
ER # 66.62 # QA_er_lr=3e-5_ep=10_rs=32_rf=3_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
MaxLoss# 66.55 # QA_mir_lr=3e-5_ep=10_rs=32_rf=3_mcs=256_largest_afterloss_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
OnL2Reg # 65.09 # qa_simplecl_lr=3e-5_ep=10_l2w=1_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
OnEWC # 65.31 # qa_oewc_lr=3e-5_ep=10_lbd=250_gm=9e-1_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
"""
"""
*Frozen # 45.77 # QA_nonecl_T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8_result.json
"""
import os 
import json
import pandas as pd

from cmr.notebooks.draw_utils import draw_stacked_bars, draw_curve

# os.chdir("experiments/results/qa/")
all_data = []
for line in maps.splitlines():
    name, OECT, path = [i.strip() for i in line.split("#")]
    # print(name, OECT, path)    
    o = json.load(open("experiments/results/qa/" + path))
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
all_data = all_data.drop(columns=["before_eval_results", "before_error_ids", "mir_buffer_ids", "retrieved_ids", "OKR_sampled_ids"])
all_data = all_data.dropna()
all_data['OEC'] = all_data.drop(columns=["timecode", "EFR", "SR", "Overall"]).mean(numeric_only=True, axis=1)
# print(all_data)
# exit()
# print(all_data)

# fig = draw_curve(df=all_data[all_data["EFR"].notnull()], fig_title=f"EFR", y_scale=[0.7, 1], x_key="timecode:O", y_key="EFR:Q", y_title="EFR")
# fig.save('figures/curves/EFRs.png', scale_factor=2.0)


color_dom = ["CFT", "OnL2Reg", "OnEWC",  "ER", "MaxLoss", "MIR"]
color_range = ['gray', 'brown',  '#7fc97f', '#D35400', 'purple', '#386cb0']

fig = draw_curve(df=all_data[all_data["UKR"].notnull()], fig_title=f"UKR", y_scale=[0.66, 0.82],  x_key="timecode:O", y_key="UKR:Q", y_title="UKR", height=600, width=500, orient="none", color_dom=color_dom, color_range=color_range)
fig.save('figures/curves/UKRs.png', scale_factor=3.0)

fig = draw_curve(df=all_data[all_data["OKR"].notnull()], fig_title=f"OKR", y_scale=[0.77, 0.96],  x_key="timecode:O", y_key="OKR:Q", y_title="OKR", height=600, width=500, orient="none", color_dom=color_dom, color_range=color_range)
fig.save('figures/curves/OKRs.png', scale_factor=3.0)

fig = draw_curve(df=all_data[all_data["CSR"].notnull()], fig_title=f"CSR", y_scale=[0.52, 0.67],  x_key="timecode:O", y_key="CSR:Q", y_title="CSR", height=600, width=500, orient="none", color_dom=color_dom, color_range=color_range)
fig.save('figures/curves/CSRs.png', scale_factor=3.0)

fig = draw_curve(df=all_data[all_data["KG"].notnull()], fig_title=f"KG", y_scale=[0.43, 0.54],  x_key="timecode:O", y_key="KG:Q", y_title="KG", height=600, width=500, orient="none", color_dom=color_dom, color_range=color_range)
fig.save('figures/curves/KGs.png', scale_factor=3.0)

fig = draw_curve(df=all_data[all_data["OEC"].notnull()], fig_title=f"OEC", y_scale=[0.61, 0.72],  x_key="timecode:O", y_key="OEC:Q", y_title="OEC", height=600, width=500, orient="none", color_dom=color_dom, color_range=color_range)
fig.save('figures/curves/OECs.png', scale_factor=3.0)