import numpy as np 
import pandas as pd
import json 
import os
import altair as alt 
import warnings
warnings.filterwarnings('ignore')
import numpy as np  

def get_all_data(path, prefix, add_baselines=False):
    with open(path) as f:
        data = json.load(f)
    online_eval_results = data["online_eval_results"]
    all_data = []
    num_all_errors = 0
    num_all_errors_baseline = 0
    MA_instant_unseen_EM = []
    MA_instant_seen_EM = []
    MA_instant_unseen_EM_basline = []
    MA_instant_seen_EM_basline = []
    MA_instant_efr = []
    for item in online_eval_results:
        dp = {}
        dp["timecode"] = item["timecode"]
        dp["method"] = prefix
        dp["instant_unseen_EM"] = item["before_eval"]["metric_results"]["EM"] 
        MA_instant_unseen_EM.append(dp["instant_unseen_EM"])
        dp["instant_unseen_EM_MA"] = np.mean(MA_instant_unseen_EM)


        dp["instant_seen_EM"] = item["after_eval"]["metric_results"]["EM"] 
        MA_instant_seen_EM.append(dp["instant_seen_EM"])
        dp["instant_seen_EM_MA"] = np.mean(MA_instant_seen_EM)

        dp["instant_unseen_EM_delta"] = item["before_eval"]["metric_results"]["EM"] - item["model0_instant_EM"]

        dp["instant_efr"] = item["instant_fixing_rate"] 
        MA_instant_efr.append(dp["instant_efr"])
        dp["instant_efr_MA"] = np.mean(MA_instant_efr)

        num_all_errors += len(item["error_ids"])
        dp["num_errors"] = num_all_errors
        all_data.append(dp)       

        # dp = {}
        # dp["timecode"] = item["timecode"]
        # dp["method"] = prefix + "_post" 
        # dp["instant_EM"] = item["after_eval"]["metric_results"]["EM"] 
        # all_data.append(dp) 

        if add_baselines:
            dp = {}
            dp["timecode"] = item["timecode"]
            dp["method"] = "baseline"
            dp["instant_unseen_EM"] = item["model0_instant_EM"]
            dp["instant_seen_EM"] = item["model0_instant_EM"]
            MA_instant_unseen_EM_basline.append(dp["instant_unseen_EM"])
            dp["instant_unseen_EM_MA"] = np.mean(MA_instant_unseen_EM_basline)
            dp["instant_seen_EM_MA"] = np.mean(MA_instant_unseen_EM_basline)
            dp["instant_efr"] = 0.0
            dp["instant_efr_MA"] = 0.0
            dp["instant_unseen_EM_delta"] = 0.0
            num_all_errors_baseline += 32 * (1-item["model0_instant_EM"])
            dp["num_errors"] = num_all_errors_baseline
            all_data.append(dp) 

        # if "accumulative_EM" in item:
        #     dp = {}
        #     dp["timecode"] = item["timecode"]
        #     dp["method"] = prefix
        #     dp["accumulative_EM"] = item["accumulative_EM"]
        #     all_data.append(dp)
        #     if add_baselines:
        #         dp = {}
        #         dp["timecode"] = item["timecode"]
        #         dp["method"] = "baseline"
        #         dp["accumulative_EM"] = item["doing-nothing_accmulative_EM"]
        #         all_data.append(dp) 
    # print(all_data_df)
    return all_data
 


def draw_curve(df, y_scale=[0, 1], fig_title="", y_title="EM", y_key="em:Q", height=800):
    x = alt.X("timecode", type="ordinal", title="Timecode")
    # y_em = alt.Y("em", type="quantitative", title="EM", scale=alt.Scale(domain=[0.5, 1.0]))
    # em_line = alt.Chart(df).mark_line(interpolate='natural', point=True).encode(x=x, y=y_em, opacity=alt.value(0.8), color=alt.value('red'))
    # scale = alt.Scale(domain=list(set(df["prefix"])), range=['red', 'orange', 'purple', 'blue', 'green'])
    # color=alt.Color('prefix:N', scale=scale)

    color=alt.Color('method:N')


    
    # fig = alt.Chart(reformatted_data).mark_area(opacity=0.6).encode(x="timecode:O", y=alt.Y("em:Q", stack=None, title="EM"), color=color)
    
    # fig = alt.Chart(df).mark_lin5(o0.8acity=0.7, interpolate="natural", point=True).encode(x=x, y=alt.Y("em:Q", stack=None, title="EM", scale=alt.Scale(domain=[0.4, 1])), color=color).properties(title="Knowledge Retain in EM acc. (forgetting measure) ")



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




# all_data = get_all_data(path="exp_results/dynamic_stream/cl_simple/nq_dev_0723_dynamic_simplecl_result_50.json", prefix="cl_simple", add_baselines=True)
all_data = get_all_data(path="exp_results/dynamic_stream/cl_simple/nq_dev_0729_dynamic_simplecl_result.json", prefix="simple_cl", add_baselines=True)
all_data += get_all_data(path="exp_results/dynamic_stream/online_ewc/nq_dev_0729_dynamic_ewc_l500_50_result.json", prefix="online_ewc")
all_data += get_all_data(path="exp_results/dynamic_stream/memory_based/nq_dev_0729_er_result.json", prefix="sparse_er")
all_data += get_all_data(path="exp_results/dynamic_stream/memory_based/nq_dev_0729_mbpa_result.json", prefix="mbpa")
all_data += get_all_data(path="exp_results/dynamic_stream/memory_based/nq_dev_0729_mbpapp_result.json", prefix="mbpa++")



all_data_df = pd.DataFrame(all_data) 

print(all_data_df)

methods = ["cl_simple", "baseline", "online_ewc", "sparse_er", "mbpa", "mbpa++"]


# instant_em_df = all_data_df[all_data_df['instant_unseen_EM'].notna()][all_data_df["method"].isin(methods)]
draw_curve(all_data_df, fig_title="Instant Unseen EM", y_scale=[0.2, 1],  y_key="instant_unseen_EM:Q").save('figures/dynamic_instant_unseen_EM.png')

draw_curve(all_data_df, fig_title="Instant Unseen EM (delta)", y_scale=[-0.5, 0.5],  y_key="instant_unseen_EM_delta:Q").save('figures/dynamic_instant_unseen_EM_delta.png')


draw_curve(all_data_df, fig_title="Instant Unseen EM (MA)", y_scale=[0.5, 1],  y_key="instant_unseen_EM_MA:Q").save('figures/dynamic_instant_unseen_EM_MA.png')

draw_curve(all_data_df, fig_title="Instant Seen EM (MA)", y_scale=[0.5, 1],  y_key="instant_seen_EM_MA:Q").save('figures/dynamic_instant_seen_EM_MA.png')


draw_curve(all_data_df, fig_title="Instant Error Fixing Rate", y_scale=[0.55, 1],  y_key="instant_efr:Q").save('figures/instant_efr.png')

draw_curve(all_data_df, fig_title="Instant Error Fixing Rate (MA)", y_scale=[0., 1],  y_key="instant_efr_MA:Q").save('figures/instant_efr_MA.png')



draw_curve(all_data_df, fig_title="Number of Errors", y_scale=[0, 600],  y_key="num_errors:Q").save('figures/dynamic_num_of_errors.png')

# cf_prefixes = []
# for bcf in base_cf_prefixes:
#     cf_prefixes.append(bcf)
#     if bcf != "baseline":
#         cf_prefixes.append(bcf+"_pre")
#         # cf_prefixes.append(bcf+"_post")

# print(base_cf_prefixes)
 
 
# instant_em_df = all_data_df[all_data_df['instant_EM'].notna()][all_data_df["prefix"].isin(cf_prefixes)]
# draw_curve(instant_em_df, fig_title="Instant EM vs Baseline", y_scale=[0.2, 1],  y_key="instant_EM:Q").save('figures/dynamic_instant_pre_vs.png')

 
# accum_em_df = all_data_df[all_data_df['accumulative_EM'].notna()][all_data_df["prefix"].isin(cf_prefixes)]
# draw_curve(df=accum_em_df, fig_title="Accumulative EM vs Baseline", y_scale=[0.5, 1],  y_key="accumulative_EM:Q").save('figures/dynamic_accumu_pre_vs.png')

