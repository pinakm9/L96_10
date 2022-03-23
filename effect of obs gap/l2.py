import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
print(module_dir)

import numpy as np
import tables
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

data_folder = 'data'
ev_time = {0.01: 315, 0.03: 105, 0.05:63, 0.07: 45, 0.09: 35}
num_seeds = 10
gap = 1
obs_cov = 0.4
dim = 10

fig_all = plt.figure(figsize=(8, 8))
ax_all = fig_all.add_subplot(111)

for obs_gap, steps in ev_time.items():
    t = list(obs_gap * np.arange(0, steps, 1))
    # find all the relevant files:
    current = []
    for folder in os.listdir(data_folder):
        folder = data_folder + '/' + folder
        if 'obs_gap_{}'.format(obs_gap) in folder:
            current.append(folder)


    # collect data
    data_c = {'time': [], 'l2_error': []}
    for folder in current:
        data_c['time'] += t 
        hdf5 = tables.open_file(folder + '/assimilation.h5', mode='r')
        error = hdf5.root.l2_error.read() / np.sqrt(dim * obs_cov)
        data_c['l2_error'] += list(error)

    sns.lineplot(ax=ax_all, data=data_c, x='time', y='l2_error', ci='sd', label='obs gap={}'.format(obs_gap))
    
ax_all.set_ylabel(r'$\frac{\rm{error}}{\sqrt{\rm{dim}}\times\rm{obs\;std}}$', fontsize=20)
ax_all.set_xlabel('time', fontsize=15)
ax_all.set_title(r'errors with 1$\sigma\,$ band', fontsize=15)  
fig_all.savefig('plots/l2_error.png')