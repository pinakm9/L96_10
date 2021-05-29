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
dist_folder = 'enkf_dists/{}_pc_{}'.format(id_, particle_count)
cov_folder = 'cov/{}_pc_{}'.format(id_, particle_count)
ev_time = 400
gap = 4

with plt.style.context('seaborn-paper'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    for i, config in enumerate(sorted(os.listdir(config_folder))):
        config = config[:-5]
        df = pd.read_csv(dist_folder + '/{}.csv'.format(config))
        sns.lineplot(data=df, x=df['time'], y=df['sinkhorn_div'], ci=None, ax=ax, label='$\mu_{}$'.format(config.split('_')[-1]))
    """
    for i, config in enumerate(sorted(os.listdir(config_folder))):
        config = config[:-5]
        df_cov = pd.read_csv(cov_folder + '/{}.csv'.format(config))
        sns.lineplot(data=df, x=df_cov['time'], y=np.sqrt(df_cov['eigh']), ci=None, ax=ax,\
                     label='sqrt of avg largest ev for BPF {} for im '.format(particle_count) + config.split('_')[-1],\
                     linestyle='dotted')
    """
    plt.xlabel('assimilation step', fontsize=20)
    plt.ylabel('$D_\epsilon$', fontsize=20)
    plt.title('$D_\epsilon(\pi^P_n(\mu), \pi^E_n(\mu))$', fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig('plots/enkf/enkf_{}_vs_bpf_{}.png'.format(200, particle_count))