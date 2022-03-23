import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
print(module_dir)

import numpy as np
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

dist_folder = 'dists'
num_seeds = 10
gap = 1
obs_cov = 0.4
ev_time = {0.01: 315, 0.03: 105, 0.05:63, 0.07: 45, 0.09: 35}

fig_all = plt.figure(figsize=(8, 8))
ax_all = fig_all.add_subplot(111)

for obs_gap, steps in ev_time.items():
    t_c = list(obs_gap * np.arange(0, steps, 1))
    # find all the relevant files:
    current, rest = [], []
    for file in os.listdir(dist_folder):
        file = dist_folder + '/' + file
        if 'obs_gap_{}'.format(obs_gap) in file:
            current.append((file, t_c))
        else:
            gap = float(re.search("obs_gap_\d+\.\d+", file).group(0)[len('obs_gap_'):])
            t = list(gap * np.arange(0, ev_time[gap], 1))
            rest.append((file, t))

    # collect data
    data_c = {'time': [], 'sinkhorn_div': []}
    data_r = {'time': [], 'sinkhorn_div': []}
    for file, t in current:
        data_c['time'] += t 
        data_c['sinkhorn_div'] += list(pd.read_csv(file, delimiter=',')['sinkhorn_div'].to_numpy())
    #print(data_c)
    for file, t in rest:
        data_r['time'] += t 
        data_r['sinkhorn_div'] += list(pd.read_csv(file, delimiter=',')['sinkhorn_div'].to_numpy())

    # plot bands
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    sns.lineplot(ax=ax, data=data_c, x='time', y='sinkhorn_div', ci='sd', label='obs gap={}'.format(obs_gap))
    sns.lineplot(ax=ax, data=data_r, x='time', y='sinkhorn_div', ci='sd', label='the rest')
    #plt.show()
    fig.savefig('plots/obs_gap_{}_obs_cov_{}.png'.format(obs_gap, obs_cov))

    sns.lineplot(ax=ax_all, data=data_c, x='time', y='sinkhorn_div', ci='sd', label='obs gap={}'.format(obs_gap))

fig_all.savefig('plots/obs_gap_{}_obs_cov_{}.png'.format('all', obs_cov))