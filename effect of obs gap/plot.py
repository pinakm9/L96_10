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

obs_cov = 0.4



dist_folder = 'dists'
error_folder = 'data'
plot_folder = 'plots'


# collect files
dist_dict_all = {obs_gap:[] for obs_gap in [0.01, 0.03, 0.05, 0.07, 0.09]}

for file in glob.glob('{}/*.csv'.format(dist_folder)):
    obs_gap = float(re.search("obs_gap_\d+\.\d+", file).group(0)[len('obs_gap_'):])
    dist_dict_all[obs_gap].append(file)


# collect files
error_dict_all_1 = {obs_gap:[] for obs_gap in [0.01, 0.03, 0.05, 0.07, 0.09]}
error_dict_all_2 = {obs_gap:[] for obs_gap in [0.01, 0.03, 0.05, 0.07, 0.09]}

for folder in glob.glob('{}/*'.format(error_folder)):
    file = folder + '/assimilation.h5'
    obs_gap = float(re.search("obs_gap_\d+\.\d+", file).group(0)[len('obs_gap_'):])
    prior = re.search("prior_\d", file).group(0)
    if prior == 'prior_1':
        error_dict_all_1[obs_gap].append(file)
    else:
        error_dict_all_2[obs_gap].append(file)


del dist_dict_all[0.05]
del error_dict_all_1[0.05]
del error_dict_all_2[0.05]

# rate.BatchRate(dist_dict_all).plot(plot_folder, tag='obs_gap_all', obs_cov=obs_cov, ylim=(0.5, 12.0), fsize=25)
l2.BatchL2([error_dict_all_1, error_dict_all_2])\
    .plot(plot_folder, tag='obs_gap_all', obs_cov=obs_cov, ylim=(0.0, 4.0), fsize=25, labels=['unbiased', 'biased'], linestyles=['solid', 'dashed'])
# trace.BatchTr([error_dict_all_1, error_dict_all_2])\
#     .plot(plot_folder, tag='obs_gap_all', obs_cov=obs_cov, ylim=(0.0, 40.0), fsize=25, labels=['unbiased', 'biased'], linestyles=['solid', 'dashed'])
# dvl2.BatchDvL2(dist_dict_all, error_dict_all_2).plot(plot_folder, tag='obs_gap_all', obs_cov=obs_cov, ylim=(0.2, 4.0), xlim=(1.5, 11.5), fsize=25)
