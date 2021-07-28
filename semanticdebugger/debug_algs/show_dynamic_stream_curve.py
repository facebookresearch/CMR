import numpy as np 
import pandas as pd
import json 
import os
import altair as alt 
import warnings
warnings.filterwarnings('ignore')
 

def get_all_data(path, prefix, add_baselines=False):
    with open(path) as f:
        data = json.load(f)
    online_eval_results = data["online_eval_results"]
    all_data = []
    for item in online_eval_results:
        dp = {}
        dp["timecode"] = item["timecode"]
        dp["prefix"] = prefix + "_pre"
        dp["instant_EM"] = item["before_eval"]["metric_results"]["EM"] 
        all_data.append(dp)       
 
        dp = {}
        dp["timecode"] = item["timecode"]
        dp["prefix"] = prefix + "_post" 
        dp["instant_EM"] = item["after_eval"]["metric_results"]["EM"] 
        all_data.append(dp) 

        if add_baselines:
            dp = {}
            dp["timecode"] = item["timecode"]
            dp["prefix"] = "baseline"
            dp["instant_EM"] = item["doing-nothing_instant_EM"]
            all_data.append(dp) 

        if "accumulative_EM" in item:
            dp = {}
            dp["timecode"] = item["timecode"]
            dp["prefix"] = prefix
            dp["accumulative_EM"] = item["accumulative_EM"]
            all_data.append(dp)
            if add_baselines:
                dp = {}
                dp["timecode"] = item["timecode"]
                dp["prefix"] = "baseline"
                dp["accumulative_EM"] = item["doing-nothing_accmulative_EM"]
                all_data.append(dp) 
    # print(all_data_df)
    return all_data
 


def draw_curve(df, y_scale=[0, 1], fig_title="", y_title="EM", y_key="em:Q", height=800):
    x = alt.X("timecode", type="ordinal", title="Timecode")
    # y_em = alt.Y("em", type="quantitative", title="EM", scale=alt.Scale(domain=[0.5, 1.0]))
    # em_line = alt.Chart(df).mark_line(interpolate='natural', point=True).encode(x=x, y=y_em, opacity=alt.value(0.8), color=alt.value('red'))
    # scale = alt.Scale(domain=list(set(df["prefix"])), range=['red', 'orange', 'purple', 'blue', 'green'])
    # color=alt.Color('prefix:N', scale=scale)

    color=alt.Color('prefix:N')


    
    # fig = alt.Chart(reformatted_data).mark_area(opacity=0.6).encode(x="timecode:O", y=alt.Y("em:Q", stack=None, title="EM"), color=color)
    
    # fig = alt.Chart(df).mark_line(opacity=0.7, interpolate="natural", point=True).encode(x=x, y=alt.Y("em:Q", stack=None, title="EM", scale=alt.Scale(domain=[0.4, 1])), color=color).properties(title="Knowledge Retain in EM acc. (forgetting measure) ")



    fig = alt.Chart(df).mark_line(opacity=1, interpolate="natural", point=True).encode(x=x, y=alt.Y(y_key, stack=None, title=y_title, scale=alt.Scale(domain=y_scale)), color=color).properties(title=fig_title)

    fig = alt.layer(fig).resolve_scale()
    fig = fig.properties(width=1100, height=height).configure_axis(
        labelFontSize=18,
        titleFontSize=16, 
    ).configure_legend(titleFontSize=0, labelFontSize=15, orient='bottom-right', strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=5,).configure_title(
        fontSize=20,
        font='Courier',
        anchor='middle',
        orient="top", align="center",
        color='black'
    )
    return fig




all_data = get_all_data(path="exp_results/dynamic_stream/cl_simple/nq_dev_0723_dynamic_simplecl_result_50.json", prefix="cl_simple", add_baselines=True)

all_data += get_all_data(path="exp_results/dynamic_stream/cl_simple/nq_dev_0723_dynamic_ewc_l500_50_result.json", prefix="online_ewc")

all_data_df = pd.DataFrame(all_data) 

base_cf_prefixes = ["cl_simple", "baseline", "online_ewc"]
cf_prefixes = []
for bcf in base_cf_prefixes:
    cf_prefixes.append(bcf)
    if bcf != "baseline":
        cf_prefixes.append(bcf+"_pre")
        # cf_prefixes.append(bcf+"_post")

print(cf_prefixes)
 
 
instant_em_df = all_data_df[all_data_df['instant_EM'].notna()][all_data_df["prefix"].isin(cf_prefixes)]
draw_curve(instant_em_df, fig_title="Instant EM vs Baseline", y_scale=[0.2, 1],  y_key="instant_EM:Q").save('figures/dynamic_instant_pre_vs.png')

 
accum_em_df = all_data_df[all_data_df['accumulative_EM'].notna()][all_data_df["prefix"].isin(cf_prefixes)]
draw_curve(df=accum_em_df, fig_title="Accumulative EM vs Baseline", y_scale=[0.5, 1],  y_key="accumulative_EM:Q").save('figures/dynamic_accumu_pre_vs.png')

