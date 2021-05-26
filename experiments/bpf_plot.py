# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# first 10 odd primes as random seeds
seeds = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
particle_count_0 = 250
particle_count_1 = 1000
id_ = 0
config_folder = '../configs/{}_pc_{}'.format(id_, particle_count_0)
dist_folder = 'bpf_dists/{}'.format(id_)
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
        sns.lineplot(data=df, x=df['time'], y=df['sinkhorn_div'], ci=None, ax=ax, label='initial measure ' + config.split('_')[-1])

    plt.xlabel('assimilation step', fontsize=20)
    plt.ylabel('$\sqrt{S_{0.01}}$', fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('plots/enkf/bpf{}_vs_bpf{}.png'.format(particle_count_0, particle_count_1))