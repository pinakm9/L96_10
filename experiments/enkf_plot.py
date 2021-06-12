# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import batch_expr as be 
import os
import numpy as np
import tables
import wasserstein as ws
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# first 10 odd primes as random seeds
particle_count = 250
id_ = 0
config_folder = '../configs/{}_pc_{}'.format(id_, particle_count)
results_folder = 'results/{}_pc_{}'.format(id_, particle_count)
enkf_folder = 'enkf_results'
ee_dist = 'ee_dists/50_vs_200'
dist_folder = 'enkf_dists/{}_pc_{}'.format(id_, 2000)
bpf_dist_folder = 'bpf_dists/{}'.format(id_)
cov_folder = 'cov/{}_pc_{}'.format(id_, particle_count)
ev_time = 400
gap = 4
#colors = ['royalblue', 'darkorange', 'darkgreen', 'r']

with plt.style.context('seaborn-paper'):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)
    for i, config in enumerate(sorted(os.listdir(config_folder))):
        config = config[:-5]
        df = pd.read_csv(dist_folder + '/{}.csv'.format(config))
        sns.lineplot(data=df, x=df['time'], y=df['sinkhorn_div'], ci=None, color=colors[i], ax=ax, label='i={}'.format(config.split('_')[-1]))
        #, label='i={}, X=E, N=200'.format(config.split('_')[-1]))
    
    for i, config in enumerate(sorted(os.listdir(config_folder))):
        config = config[:-5]
        y = np.average(np.load(ee_dist + '/{}.npy'.format(config)), axis=1)
        x = list(range(0, ev_time, gap))
        data = {'time': x, 'sinkhorn_div': y}
        df = pd.DataFrame(data)
        sns.lineplot(data=df, x=df['time'], y=df['sinkhorn_div'], ci=None, color=colors[i], ax=ax, linestyle='dashed')


    for i, config in enumerate(sorted(os.listdir(config_folder))):
        config = config[:-5]
        df_pf = pd.read_csv(bpf_dist_folder + '/{}_{}_{}_vs_2000.csv'.format(id_, config, particle_count))
        sns.lineplot(data=df_pf, x=df_pf['time'], y=df_pf['sinkhorn_div'], ci=None, color=colors[i], ax=ax,\
                     #label='i={}, X=P, N=250'.format(config.split('_')[-1]),\
                     linestyle='dotted')
    
    plt.xlabel('assimilation step (n)', fontsize=30)
    plt.ylabel('$D_\epsilon$', fontsize=30)
    plt.title('$L96 (10),  D_\epsilon(\pi^{X, N}_n(\mu_i), \pi^{Y, M}_n(\mu_i))$', fontsize=40)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('plots/enkf/distance between different filters.png'.format(200, particle_count))