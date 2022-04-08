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
import l2

obs_cov = 0.4



data_folder = 'data'
plot_folder = 'plots'


# collect files
file_dict_all = {obs_gap:{'prior_1': [], 'prior_3': []} for obs_gap in [0.01, 0.03, 0.05, 0.07, 0.09]}

for folder in glob.glob('{}/*'.format(data_folder)):
    file = folder + '/assimilation.h5'
    obs_gap = float(re.search("obs_gap_\d+\.\d+", file).group(0)[len('obs_gap_'):])
    prior = re.search("prior_\d", file).group(0)
    file_dict_all[obs_gap][prior].append(file)


for gap in file_dict_all:
    err_1 = l2.L2Error(file_dict_all[gap]['prior_1'], gap)
    err_1.collect_mean_data()
 
    err_3 = l2.L2Error(file_dict_all[gap]['prior_3'], gap)
    err_3.collect_mean_data() 


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    phy_time, l2_error = err_1.collect_data()
    #ax.scatter(phy_time, l2_error, s=3, c='grey', alpha=0.3)
    phy_time, l2_error = err_3.collect_data()
    #ax.scatter(phy_time, l2_error, s=3, c='grey', alpha=0.3)
    ax.plot(err_1.phy_time, np.ones_like(err_1.phy_time) * np.sqrt(obs_cov), label='$\sigma$')
    ax.plot(err_1.phy_time, err_1.l2_error, c='black', label=r'mean error prior 1', linestyle='solid')
    ax.plot(err_3.phy_time, err_3.l2_error, c='black', label=r'mean error prior 2', linestyle='dotted')
    ax.set_ylabel(r'$\frac{\rm error}{\sqrt{\rm dimension}}$', fontsize=20)
    ax.set_xlabel(r'time ($t=ng$)', fontsize=20)
    ax.set_title(r'g = {:.2f}, $\sigma$ = {:.2f}'.format(err_1.obs_gap, obs_cov), fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('{}/l2_error_obs_gap_{:.2f}.png'.format(plot_folder, gap))