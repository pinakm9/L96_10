import sys
from pathlib import Path
from os.path import dirname, abspath
from cv2 import threshold
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
print(module_dir)

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os, re
import utility as ut


obs_cov = 0.4



dist_folder = 'dists'
error_folder = 'data'
plot_folder = 'plots'
ev_time = 63
obs_gap = 0.05
phy_time = np.array(range(ev_time)) * obs_gap
num_seeds = 10
fsize=20
folders = ['dists/1', 'dists/01', 'dists/001', 'dists/0001']
styles = ['dashed', 'solid', 'dotted', 'dashdot']
labels = [r'$\varepsilon = 0.1$', r'$\varepsilon = 0.01$', r'$\varepsilon = 0.001$', r'$\varepsilon = 0.0001$',]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
for i, folder in enumerate(folders):
    dist = np.zeros(ev_time)
    for filename in os.listdir(folder):
        file = '{}/{}'.format(folder, filename)
        dist += pd.read_csv(file, delimiter=',')['sinkhorn_div'].to_numpy()
    dist /= num_seeds
    ax.plot(phy_time, dist, linestyle=styles[i], label=labels[i], color='black')
ax.legend(fontsize=fsize-0, loc='upper right')
ax.set_ylabel(r'$\mathbb{E}[D_\varepsilon\left(\pi_n(\mu_0), \pi_n(\mu_b)\right)]$', fontsize=fsize+10)
ax.set_xlabel(r'time ($t=ng$)', fontsize=fsize+10)
# ax.set_title(r'$g = {:.2f},\,\sigma^2= {:.2f}$'.format(self.rcs[i].obs_gap, obs_cov), fontsize=fsize)
fig.savefig('{}/eps_all.png'.format(plot_folder), dpi=300, bbox_inches='tight', pad_inches=0)