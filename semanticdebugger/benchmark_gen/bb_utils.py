import random, copy
from scipy.stats import betabinom
import numpy as np

def bb_sample(n, a, b, prefix="bb", count=1):
    n = int(n)
    b = float(b)
    mean, var, skew, kurt = betabinom.stats(n, a, b, moments='mvsk')
    x = np.arange(betabinom.ppf(1e-18, n, a, b),
                                betabinom.ppf(1-1e-18, n, a, b))
    bb = betabinom.pmf(x, n, a, b)
    bb = bb * count
    pmf_pd = []
    for time_step, prob in enumerate(list(bb)):
        pmf_pd.append(dict(time_step=time_step, p=prob, prefix=prefix))
    return pmf_pd
    
def bb_rescale(all_pmfs, batch_size):
    scaled_data = {}
    for item in all_pmfs:
        t = item["time_step"]
        if t not in scaled_data:
            scaled_data[t] = {}
        scaled_data[t][item["prefix"]] = item["p"]
    
    for t in scaled_data:
        prefixes = list(scaled_data[t].keys())
        sum_values = sum(list(scaled_data[t].values()))
        scale_ratio = batch_size/sum_values
        random.shuffle(prefixes)
        p_pool = []
        for p in prefixes:
            scaled_data[t][p] *= scale_ratio
            scaled_data[t][p] = round(scaled_data[t][p])
            if scaled_data[t][p] > 5:
                p_pool.append(p)
        buffer = sum(list(scaled_data[t].values())) - batch_size
        if buffer != 0:
            picked_p = random.choice(p_pool)
            scaled_data[t][picked_p] -= buffer 
        assert sum(list(scaled_data[t].values())) == batch_size

    return_all_pmfs = copy.deepcopy(all_pmfs)
    for item in return_all_pmfs:
        item["p"] = int(scaled_data[item["time_step"]][item["prefix"]])
    return return_all_pmfs


def build_submission_stream(submission_data, scaled_all_pmfs, config, args):
    submission_stream = []
    scaled_data = {}
    for item in scaled_all_pmfs:
        t = item["time_step"]
        if t not in scaled_data:
            scaled_data[t] = {}
        scaled_data[t][item["prefix"]] = int(item["p"])
    num_episodes = len(scaled_data)
    assert num_episodes ==  args.num_episodes
    data_names = list(submission_data.keys())
    init_error_pmfs = []
    for t in range(num_episodes):
        episode = []
        for data_name in data_names:
            _data_name = data_name
            if config[data_name]["upstream"]:
                _data_name = f"*{data_name}"
            if _data_name in scaled_data[t]:
                _sampled = random.sample(submission_data[data_name], k=scaled_data[t][_data_name])
                episode += _sampled
                num_errors = len([1 for item in _sampled if item["init_status"]=="error"])
                init_error_pmfs.append(dict(time_step=t, p=num_errors, prefix=_data_name))
        assert len(episode) ==  args.episode_size
        submission_stream.append(episode)

    return submission_stream, init_error_pmfs