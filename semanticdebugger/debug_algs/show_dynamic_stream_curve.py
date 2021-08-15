import numpy as np 
import pandas as pd
import json 
import os
import altair as alt 
import warnings
warnings.filterwarnings('ignore')
import numpy as np  

def get_all_data(path, prefix):
    with open(path) as f:
        data = json.load(f)
    online_eval_results = data["online_eval_results"]
    all_data = []
    num_all_errors = 0 
    MA_instant_unseen_EM = []
    MA_instant_seen_EM = [] 
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

        dp["instant_efr"] = item["instant_fixing_rate"] 
        MA_instant_efr.append(dp["instant_efr"])
        dp["instant_efr_MA"] = np.mean(MA_instant_efr)

        num_all_errors += len(item["error_ids"])
        dp["num_errors"] = num_all_errors
        all_data.append(dp)       

    return all_data
 

def get_revisited_data(path):
    with open(path) as f:
        data_stream = json.load(f) 
    num_exp = 0
    seen_ids = set()
    num_revisited = 0
    all_data = []
    for timecode, episode in enumerate(data_stream):
        dp = {}
        dp["timecode"] = timecode
        for item in episode:
            _id = item["id"]
            if _id not in seen_ids:
                seen_ids.add(_id)
            else:
                num_revisited += 1
        dp["num_revisited"] = num_revisited
        all_data.append(dp)
        num_exp += len(episode)
    for dp in all_data:
        dp["revisited_ratio"] = dp["num_revisited"] / num_exp
    return all_data



def draw_curve(df, y_scale=[0, 1], fig_title="", y_title="EM", y_key="em:Q", height=800, width=1100):
    x = alt.X("timecode", type="ordinal", title="Timecode", axis=alt.Axis(tickCount=df.shape[0]//10, grid=False) )  
    color=alt.Color('method:N', sort=["none_cl"]) 
    fig = alt.Chart(df).mark_line(opacity=1, interpolate="natural", point=True).encode(x=x, y=alt.Y(y_key, stack=None, title=y_title, scale=alt.Scale(domain=y_scale)), color=color).properties(title=fig_title)

    fig = alt.layer(fig).resolve_scale()
    fig = fig.properties(width=width, height=height).configure_axis(
        labelFontSize=13,
        titleFontSize=16, 
    ).configure_legend(titleFontSize=0, labelFontSize=15, orient='bottom-right', strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=5,).configure_title(
        fontSize=20,
        font='Courier',
        anchor='middle',
        orient="top", align="center",
        color='black',
    )
    return fig




# all_data = get_all_data(path="exp_results/dynamic_stream/cl_simple/nq_dev_0723_dynamic_simplecl_result_50.json", prefix="cl_simple", add_baselines=True)
all_data = get_all_data(path="exp_results/dynamic_stream/none/nq_dev_0813_wr_wpara_dynamic_none_result.json", prefix="none_cl")
all_data += get_all_data(path="exp_results/dynamic_stream/cl_simple/nq_dev_0813_wr_wpara_dynamic_simplecl_result.json", prefix="simple_cl")
all_data += get_all_data(path="exp_results/dynamic_stream/online_ewc/nq_dev_0813_wr_wpara_dynamic_ewc_result.json", prefix="online_ewc")
all_data += get_all_data(path="exp_results/dynamic_stream/memory_based/nq_dev_0813_wr_wpara_er_result.json", prefix="sparse_er")
all_data += get_all_data(path="exp_results/dynamic_stream/memory_based/nq_dev_0813_wr_wpara_mbpa_result.json", prefix="mbpa")
all_data += get_all_data(path="exp_results/dynamic_stream/memory_based/nq_dev_0813_wr_wpara_mbpapp_result.json", prefix="mbpa++")

revisited_data = get_revisited_data(path="exp_results/data_streams/mrqa_naturalquestions_dev.data_stream.test.wr.wpara.json")


all_data_df = pd.DataFrame(all_data) 
revisited_data_df = pd.DataFrame(revisited_data)

# print(all_data_df)


methods = [ "none", "cl_simple", "online_ewc", "sparse_er", "mbpa", "mbpa++"]
# methods = ["cl_simple", "baseline", "online_ewc"]


draw_curve(revisited_data_df, fig_title="Ratio of Revisited Examples", y_scale=[0, 0.45],  y_key="revisited_ratio:Q", width=500, height=500).save('figures/revisited_ratio.png')

# instant_em_df = all_data_df[all_data_df['instant_unseen_EM'].notna()][all_data_df["method"].isin(methods)]
draw_curve(all_data_df, fig_title="Instant EM", y_scale=[0.35, 1],  y_key="instant_unseen_EM:Q", width=1000).save('figures/dynamic_instant_unseen_EM.png')

# draw_curve(all_data_df, fig_title="Instant EM (delta)", y_scale=[-0.5, 0.5],  y_key="instant_unseen_EM_delta:Q").save('figures/dynamic_instant_unseen_EM_delta.png')

draw_curve(all_data_df, fig_title="Instant EM (MA)", y_scale=[0.55, 0.8],  y_key="instant_unseen_EM_MA:Q", width=1000).save('figures/dynamic_instant_unseen_EM_MA.png')

draw_curve(all_data_df, fig_title="Instant Seen EM (MA)", y_scale=[0.5, 1],  y_key="instant_seen_EM_MA:Q").save('figures/dynamic_instant_seen_EM_MA.png')


draw_curve(all_data_df, fig_title="Instant Error Fixing Rate", y_scale=[0.55, 1],  y_key="instant_efr:Q").save('figures/instant_efr.png')

draw_curve(all_data_df, fig_title="Instant Error Fixing Rate (MA)", y_scale=[0., 1],  y_key="instant_efr_MA:Q").save('figures/instant_efr_MA.png')



draw_curve(all_data_df, fig_title="Number of Errors", y_scale=[0, 1300],  y_key="num_errors:Q", width=750, height=800).save('figures/dynamic_num_of_errors.png')

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

