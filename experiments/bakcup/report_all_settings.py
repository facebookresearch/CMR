# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pandas as pd
import os 
import glob 
from io import StringIO
import altair as alt
from torch import clamp

from cmr.notebooks.draw_utils import draw_grouped_bars

# os.chdir()
# os.makedirs("csvs/", exist_ok=True)
# result_files = []
all_data = []
header = ""
for file in glob.glob("experiments/results/qa/csvs/*.csv"):
    print(file)
    lines = open(file).read().splitlines()
    header = lines[0]
    data = lines[1:]
    all_data += data
    # result_files.append(file)
    # data = pd.read_csv(file)  
# print(all_data)
# print(len(all_data))
all_data.insert(0, header)

df = pd.read_csv(StringIO("\n".join(all_data)))

cl_methods = ['none_cl', 'simple_cl-l2w=0.0', 'online_ewc-250.0-0.9', 'er-32-3-l2w=0.0', 'mir-32/256-3-none-l2w=0.0', 'mir-32/256-3-largest_afterloss-l2w=0.0']
cl_prefix = ["Frozen", "CFT", "OnEWC", "ER", "MIR", "MaxLoss"]
for a,b in zip(cl_methods, cl_prefix):
    df = df.replace(a, b)

df.rename(columns={'OEC(T)':'OECT'}, inplace=True)
# df = df[df.cl_method != "Frozen"]
settings = [(0.9, 0.5, 0.8), (0.9, 0.1, 0.8), (0.9, 0.9, 0.8), (0.9, 0.5, 0.2), (0.9, 0.5, 0.5), (0.1, 0.5, 0.8)]
# (0.9, 0.1, 0.8), (0.9, 0.9, 0.8)

table = []

for alpha, beta, gamma in settings:
    data = df[(df["alpha"]==alpha) & (df["beta"]==beta) & (df["gamma"]==gamma)] 
    prefix = f"$alpha$={alpha},$beta$={beta},$gamma$={gamma}"
    # print()
    OECTs = {c: data[data.cl_method==c].iloc[0]["OECT"] for c in cl_prefix[:]}
    OECTs["prefix"] = prefix
    table.append(OECTs)
    # print(data)
    # print(OECTs)
    # color_dom = cl_prefix
    # color_range = ["gray", "blue", "orange", "green", "black"]
    # fig = draw_grouped_bars(df=data, fig_title=f"{alpha}-{beta}-{gamma}", y_scale=[0.6, 0.68],  x_key="cl_method", y_key="OECT", y_title="", height=250, width=175, color_dom=color_dom, color_range=color_range, bin_width=30)
    # color=alt.Color("cl_method", scale=alt.Scale(domain=color_dom, range=color_range), sort=color_dom, legend=None)  
    # print(data)
    # y=alt.Y("OECT:Q", scale=alt.Scale(domain=[0.3, 0.7]), axis=alt.Axis(grid=False))
    # fig = alt.Chart(data).mark_bar(clip=True).encode(x="cl_method",  y=y)
    # fig.save(f'figures/settings/{alpha}-{beta}-{gamma}.png', scale_factor=3.0)
# fig.save("")
table = pd.DataFrame(table)
print(table.to_csv(index=False,))