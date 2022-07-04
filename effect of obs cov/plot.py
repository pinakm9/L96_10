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
import rate, l2, trace, dvl2

obs_gap = 0.05



dist_folder = 'dists'
error_folder = 'data'
plot_folder = 'plots'


# collect files
dist_dict_all = {obs_cov:[] for obs_cov in [0.2, 0.4, 0.8, 1.6]}

for file in glob.glob('{}/*.csv'.format(dist_folder)):
    obs_cov = float(re.search("obs_cov_\d+\.\d+", file).group(0)[len('obs_cov_'):])
    dist_dict_all[obs_cov].append(file)


# collect files
error_dict_all_1 = {obs_cov:[] for obs_cov in [0.2, 0.4, 0.8, 1.6]}
error_dict_all_2 = {obs_cov:[] for obs_cov in [0.2, 0.4, 0.8, 1.6]}

for folder in glob.glob('{}/*'.format(error_folder)):
    file = folder + '/assimilation.h5'
    obs_cov = float(re.search("obs_cov_\d+\.\d+", file).group(0)[len('obs_cov_'):])
    prior = re.search("prior_\d", file).group(0)
    if prior == 'prior_1':
        error_dict_all_1[obs_cov].append(file)
    else:
        error_dict_all_2[obs_cov].append(file)

rate.BatchRate2(dist_dict_all, obs_gap).plot(plot_folder, tag='obs_cov_all', ylim=(0.5, 12.0), fsize=40, linewidth=5)
l2.BatchL22([error_dict_all_1, error_dict_all_2], obs_gap)\
   .plot(plot_folder, tag='obs_cov_all', ylim=(0.0, 4.0), fsize=40, labels=['unbiased', 'biased'], linestyles=['solid', 'dashed'], linewidth=5)
trace.BatchTr2([error_dict_all_1, error_dict_all_2], obs_gap)\
    .plot(plot_folder, tag='obs_cov_all', ylim=(0.0, 20.0), fsize=40, labels=[r'unbiased', 'biased'], linestyles=['solid', 'dashed'])
dvl2.BatchDvL22(dist_dict_all, error_dict_all_2, obs_gap).plot(plot_folder, tag='obs_cov_all', ylim=(0.2, 4.0), xlim=(1.5, 11.5), fsize=40)