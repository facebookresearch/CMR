# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from cmr.notebooks.draw_utils import draw_stacked_bars

from altair.vegalite.v4.schema.core import ColorName
from sklearn.utils import validation
import pandas as pd
import json

def visualize_stream(submission_stream, data_names, cfg):
    task_name = cfg["task_name"]
    episode_size = cfg["b"]
    submission_stat = []
    init_error_stat = []
    for time_step, episode_data in enumerate(list(submission_stream)):
        for dn in data_names:
            examples = [ex for ex in episode_data if ex["data_name"]==dn]
            num_init_errors = [ex for ex in examples if ex["init_status"]=="error"]
            if dn == data_names[0]:
                dn = "*" + dn 
            submission_stat.append(dict(time_step=time_step, num_examples=len(examples), prefix=dn))
            init_error_stat.append(dict(time_step=time_step, num_examples=len(num_init_errors), prefix=dn))
            
    submission_stat_pd = pd.DataFrame(submission_stat)
    filename_str = f"T={cfg['T']},b={cfg['b']},alpha={cfg['alpha']},beta={cfg['beta']},gamma={cfg['gamma']}|[{cfg['stream_id']}]"
    title_str = f"alpha={cfg['alpha']}, beta={cfg['beta']}, gamma={cfg['gamma']}"
    fig1 =  draw_stacked_bars(df=submission_stat_pd, fig_title=f"Submission Stream ({title_str})", y_scale=[0., episode_size+1], x_key="time_step", y_key="sum(num_examples)", y_title="# of Examples")
    fig1.save(f'figures/{task_name}.submission.{filename_str}.png', scale_factor=2.0)
    init_error_stat_pd = pd.DataFrame(init_error_stat)
    fig2 =  draw_stacked_bars(df=init_error_stat_pd, fig_title=f"(Initial) Error Stream ({title_str})", y_scale=[0., episode_size+1], x_key="time_step", y_key="sum(num_examples)", y_title="# of Errors")
    fig2.save(f'figures/{task_name}.init_error.{filename_str}.png', scale_factor=2.0)    
    
    # 50-version
    # color_dom = ["*squad", "hotpot", "news", "nq", "search", "trivia"]
    # color_range = ["gray", "blue", "orange", "green", "black", "brown"]
    # color_range = ['#bab0ac', '#f0027f',  '#7fc97f', '#D35400', '#9c9ede', '#386cb0']

    # color_dom=None; color_range=None
    # fig1 =  draw_stacked_bars(df=submission_stat_pd[submission_stat_pd["time_step"]<=50], x_scale=[0, 50], fig_title=f"Submission Stream ({title_str})", y_scale=[0., 65], x_key="time_step", y_key="sum(num_examples)", y_title="# of Examples", width=1000, bin_width=18, color_dom=color_dom, color_range=color_range)
    # fig1.save(f'figures/{task_name}.submission.{filename_str}.50.png', scale_factor=2.0)
    # init_error_stat_pd = pd.DataFrame(init_error_stat)
    # fig2 =  draw_stacked_bars(df=init_error_stat_pd[init_error_stat_pd["time_step"]<=50], x_scale=[0, 50], fig_title=f"(Initial) Error Stream ({title_str})", y_scale=[0., 65], x_key="time_step", y_key="sum(num_examples)", y_title="# of Errors", width=1000, bin_width=18, color_dom=color_dom, color_range=color_range)
    # fig2.save(f'figures/{task_name}.init_error.{filename_str}.50.png', scale_factor=2.0)    
    return 

if __name__ == '__main__': 
    qa_data_names = ["squad", "nq", "trivia", "hotpot", "news", "search"]
    cfg = dict(task_name="qa", upstream="squad") # T=100, b=64, alpha=0.9, beta=0.5, gamma=0.8
    with open("experiments/eval_data/qa/submission_stream.T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8-test.json") as f:
        streams = json.load(f)
        str_start = f.name.index("submission_stream.") + len("submission_stream.")
        str_end = f.name.index("-")
        ns_config_str = f.name[str_start:str_end]
    ns_config = eval(f"dict({ns_config_str})")
    cfg.update(ns_config)
    print(cfg)

    for stream_id, stream in enumerate(streams):
        cfg["stream_id"] = stream_id
        visualize_stream(stream, qa_data_names, cfg)
