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
import seaborn as sns
import matplotlib.pyplot as plt
import os, re
from sklearn.linear_model import LinearRegression
import utility as ut
import trace as tr

obs_cov = 0.4



data_folder = 'data'
plot_folder = 'plots'


# collect files
# collect files
file_dict_all = {obs_gap:{'prior_1': [], 'prior_3': []} for obs_gap in [0.01, 0.03, 0.05, 0.07, 0.09]}

for folder in glob.glob('{}/*'.format(data_folder)):
    file = folder + '/assimilation.h5'
    obs_gap = float(re.search("obs_gap_\d+\.\d+", file).group(0)[len('obs_gap_'):])
    prior = re.search("prior_\d", file).group(0)
    file_dict_all[obs_gap][prior].append(file)


for gap in file_dict_all:
    trace_1 = tr.Trace(file_dict_all[gap]['prior_1'], gap)
    trace_1.collect_mean_data()
 
    trace_3 = tr.Trace(file_dict_all[gap]['prior_3'], gap)
    trace_3.collect_mean_data() 


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    phy_time, trace = trace_1.collect_data()
    #ax.scatter(phy_time, trace, s=3, c='grey', alpha=0.3)
    phy_time, trace = trace_3.collect_data()
    #ax.scatter(phy_time, l2_error, s=3, c='grey', alpha=0.3)
    ax.plot(trace_1.phy_time, trace_1.trace, c='black', label=r'mean trace prior 1', linestyle='solid')
    ax.plot(trace_3.phy_time, trace_3.trace, c='black', label=r'mean error prior 2', linestyle='dotted')
    ax.set_ylabel(r'mean posterior trace', fontsize=20)
    ax.set_xlabel(r'time ($t=ng$)', fontsize=20)
    ax.set_title(r'g = {:.2f}, $\sigma$ = {:.2f}'.format(trace_1.obs_gap, obs_cov), fontsize=20)
    ax.set_ylim(2., 40.)
    plt.legend(fontsize=20)
    plt.savefig('{}/trace_obs_gap_{:.2f}.png'.format(plot_folder, gap))