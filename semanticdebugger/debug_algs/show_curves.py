import numpy as np 
import pandas as pd
import json 
import os
import altair as alt 
import warnings
warnings.filterwarnings('ignore')

def get_prefix(filepath):
    return filepath.split("/")[2].replace("_offline_eval","").replace("nq_dev_", "")[5:]

def get_result_data(path):
    # lr = path.split("_")[-5]
    # num_epoch = path.split("_")[-4]
    assert os.path.exists(path)
    output_info = json.load(open(path))
    # prefix = f"{lr};{num_epoch}"
    prefix = get_prefix(path)
    # print(output_info.keys()) 
    online_debug_results = output_info
    return prefix, online_debug_results

def get_forgetting_data(path, add_upperbound=False):
    prefix, online_debug_results = get_result_data(path)
    forgetting_data = []
    # em_on_passes = []
    # f1_on_passes = []
    for timecode, item in online_debug_results.items():
        timecode = int(timecode)
        r = item["eval_results_overall_forget"]["metric_results"]
        # result, result_all = item 
        # print(timecode, result["EM"])
        # d = dict(timecode=timecode, em=result["EM"], f1=result["QA-F1"])
        # pass_forgetting_data.append(d)
        # em_on_passes.append(result["EM"])
        # f1_on_passes.append(result["QA-F1"])
        forgetting_data.append(dict(prefix=prefix, timecode=timecode, em=r["EM"]))
        if add_upperbound:
            forgetting_data.append(dict(prefix="reference", timecode=timecode, em=1))    
        
    return forgetting_data


def get_overall_data_overtime(filepath, add_upper_bound=False):
    data = json.load(open(filepath))
    overall_alltime_error_fixing_rate = []
    # lr = filepath.split("_")[-5]
    # num_epoch = filepath.split("_")[-4]
    # prefix = f"{lr};{num_epoch}"
    prefix = get_prefix(filepath)
    
    for timecode, item in data.items():
        results = item["eval_results_overall_bug"]["metric_results"]
        dp = {}
        dp["timecode"] = int(timecode)
        dp["em"] = results["EM"]
        dp["f1"] = results["QA-F1"]
        dp["prefix"] = prefix
        overall_alltime_error_fixing_rate.append(dp)
        if add_upper_bound:
            dp = dp.copy()
            dp["prefix"] = "reference"
            dp["em"] = 1/len(data)*dp["timecode"]
            overall_alltime_error_fixing_rate.append(dp)
    # dp = {}
    # dp["timecode"] = 0
    # dp["em"] = 0
    # dp["prefix"] = prefix
    # overall_alltime_error_fixing_rate.append(dp)
    # dp = dp.copy()
    # dp["prefix"] = "reference"
    # overall_alltime_error_fixing_rate.append(dp)
    return overall_alltime_error_fixing_rate


def setup_data():

    forgetting_data = [] 
    overall_alltime_error_fixing_rate = []

    def _load_data(path, add_reference=False, forgetting_data=forgetting_data, overall_alltime_error_fixing_rate=overall_alltime_error_fixing_rate):
        forgetting_data += get_forgetting_data(path, add_upperbound=add_reference)
        overall_alltime_error_fixing_rate += get_overall_error_fixing_rate_overtime(filepath=path, add_upper_bound=add_reference)

    _load_data(path="bug_data/output/nq_dev_0706_3e-5_e5_offline_eval/alltime_result.json", add_reference=True)
    _load_data(path="bug_data/output/nq_dev_0706_1e-5_e5_offline_eval/alltime_result.json")
    _load_data(path="bug_data/output/nq_dev_0706_3e-5_e3_offline_eval/alltime_result.json")
    _load_data(path="bug_data/output/nq_dev_0706_1e-5_e3_offline_eval/alltime_result.json")
    #
    _load_data(path="bug_data/output/nq_dev_0708_ewc_l0.5_g1_3e-5_e5_offline_eval/alltime_result.json", add_reference=True)
    _load_data(path="bug_data/output/nq_dev_0708_ewc_l5_g1_3e-5_e5_offline_eval/alltime_result.json")
    _load_data(path="bug_data/output/nq_dev_0708_ewc_l50_g1_3e-5_e5_offline_eval/alltime_result.json")
    _load_data(path="bug_data/output/nq_dev_0708_ewc_l500_g1_3e-5_e5_offline_eval/alltime_result.json")  # the best
    _load_data(path="bug_data/output/nq_dev_0708_ewc_l5000_g1_3e-5_e5_offline_eval/alltime_result.json")
    _load_data(path="bug_data/output/nq_dev_0708_ewc_l50000_g1_3e-5_e5_offline_eval/alltime_result.json")

    _load_data(path="bug_data/output/nq_dev_0708_ewc_withup_l500_g1_3e-5_e5_offline_eval/alltime_result.json")
    _load_data(path="bug_data/output/nq_dev_0708_ewc_withup_l5000_g1_3e-5_e5_offline_eval/alltime_result.json")

 
    forgetting_data_df = pd.DataFrame(forgetting_data) 
    overall_alltime_error_fixing_rate_df = pd.DataFrame(overall_alltime_error_fixing_rate)

    return forgetting_data_df, overall_alltime_error_fixing_rate_df


def compute_f1_df(retain_df, errorfix_df, prefixes):
    rf_f1_data = [] 
    overall_f1_overtime = [] 
    for prefix in prefixes: 
        for timecode in range(0, 51):
            if timecode == 0:
                ef_acc = 0
                rt_acc = 1.0
            else:
                ef_acc = errorfix_df[errorfix_df["timecode"]==timecode][errorfix_df["prefix"]==prefix].iloc[0]["em"]
                rt_acc = retain_df[retain_df["timecode"]==timecode][retain_df["prefix"]==prefix].iloc[0]["em"]

            f1 = 2*(ef_acc*rt_acc)/(rt_acc+ef_acc)
            dp = {}
            dp["prefix"] = prefix
            dp["timecode"] = timecode
            dp["ef_acc"] = ef_acc
            dp["rt_acc"] = rt_acc
            dp["f1"] = f1
            overall_f1_overtime.append(dp)
            # print(f"{f1*100:.2f}%")
    overall_f1_overtime_df = pd.DataFrame(overall_f1_overtime)
    return overall_f1_overtime_df


def draw_curve(df, y_scale=[0, 1], fig_title="", y_title="EM", y_key="em:Q", height=800):
    x = alt.X("timecode", type="ordinal", title="Timecode")
    # y_em = alt.Y("em", type="quantitative", title="EM", scale=alt.Scale(domain=[0.5, 1.0]))
    # em_line = alt.Chart(df).mark_line(interpolate='natural', point=True).encode(x=x, y=y_em, opacity=alt.value(0.8), color=alt.value('red'))
    # scale = alt.Scale(domain=list(set(df["prefix"])), range=['red', 'orange', 'purple', 'blue', 'green'])
    # color=alt.Color('prefix:N', scale=scale)

    color=alt.Color('prefix:N')


    
    # fig = alt.Chart(reformatted_data).mark_area(opacity=0.6).encode(x="timecode:O", y=alt.Y("em:Q", stack=None, title="EM"), color=color)
    
    # fig = alt.Chart(df).mark_line(opacity=0.7, interpolate="natural", point=True).encode(x=x, y=alt.Y("em:Q", stack=None, title="EM", scale=alt.Scale(domain=[0.4, 1])), color=color).properties(title="Knowledge Retain in EM acc. (forgetting measure) ")



    fig = alt.Chart(df).mark_line(opacity=0.7, interpolate="natural", point=True).encode(x=x, y=alt.Y(y_key, stack=None, title=y_title, scale=alt.Scale(domain=y_scale)), color=color).properties(title=fig_title)

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


forgetting_data_df, overall_alltime_error_fixing_rate_df = setup_data()
prefixes = list(set(forgetting_data_df["prefix"]))
print(prefixes)
overall_f1_overtime_df = compute_f1_df(forgetting_data_df, overall_alltime_error_fixing_rate_df, prefixes)

# create cumulative sum
overall_f1_overtime_df["f1_cumsum"] = overall_f1_overtime_df.groupby(["prefix"])["f1"].cumsum(axis=0)

# print(overall_f1_overtime_df.head(100))

cf_prefixes = ['1e-5_e5', '3e-5_e3', '1e-5_e3', '3e-5_e5', 'reference', ]
# ewc_prefixes = ['3e-5_e5', 'ewc_l50_g1_3e-5_e5', 'ewc_l500_g1_3e-5_e5', 'ewc_l0.5_g1_3e-5_e5', 'ewc_l5_g1_3e-5_e5', 'ewc_l5000_g1_3e-5_e5', 'ewc_l50000_g1_3e-5_e5', 'reference',]
ewc_prefixes = [p for p in prefixes if "ewc" in p or "reference" in p or p=="3e-5_e5"]


draw_curve(df=forgetting_data_df[forgetting_data_df["prefix"].isin(cf_prefixes)], fig_title="Knowledge Retain in EM acc. (forgetting measure) ", y_scale=[0.4, 1]).save('figures/cf_kr_curve.png')
draw_curve(df=overall_alltime_error_fixing_rate_df[overall_alltime_error_fixing_rate_df["prefix"].isin(cf_prefixes)], fig_title="Overall Bug-Fixing Rate in EM acc. ", y_scale=[0, 1]).save('figures/cf_er_curve.png')
draw_curve(df=overall_f1_overtime_df[overall_f1_overtime_df["prefix"].isin(cf_prefixes)], fig_title="Overall F1 (retain and bugfixing). ", y_scale=[0, 1], y_title="Overall F1", y_key="f1:Q").save('figures/cf_f1_curve.png')

draw_curve(df=forgetting_data_df[forgetting_data_df["prefix"].isin(ewc_prefixes)], fig_title="Knowledge Retain in EM acc. (forgetting measure) ", y_scale=[0.5, 1]).save('figures/ewc_kr_curve.png')
draw_curve(df=overall_alltime_error_fixing_rate_df[overall_alltime_error_fixing_rate_df["prefix"].isin(ewc_prefixes)], fig_title="Overall Bug-Fixing Rate in EM acc. ", y_scale=[0, 1]).save('figures/ewc_er_curve.png')
draw_curve(df=overall_f1_overtime_df[overall_f1_overtime_df["prefix"].isin(ewc_prefixes)], fig_title="Overall F1 (retain and bugfixing). ", y_scale=[0, 1], y_title="Overall F1", y_key="f1:Q").save('figures/ewc_f1_curve.png')


draw_curve(df=overall_f1_overtime_df[overall_f1_overtime_df["prefix"].isin(ewc_prefixes)], fig_title="Overall F1-Cumsum (retain and bugfixing). ", y_scale=[0, 34], y_title="Overall F1", y_key="f1_cumsum:Q", height=1300).save('figures/ewc_f1_cumsum_curve.png')

