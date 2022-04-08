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
import rate as rc

obs_gap = 0.05



dist_folder = 'dists'
avg_dist_folder = 'avg_dists'
plot_folder = 'plots'


# collect files
file_dict_all = {obs_cov:[] for obs_cov in [0.2, 0.4, 0.8, 1.6]}
avg_file_dict_all = {}

for file in glob.glob('{}/*.csv'.format(dist_folder)):
    obs_cov = float(re.search("obs_cov_\d+\.\d+", file).group(0)[len('obs_cov_'):])
    file_dict_all[obs_cov].append(file)

for file in glob.glob('{}/*.csv'.format(avg_dist_folder)):
    obs_cov = float(re.search("obs_cov_\d+\.\d+", file).group(0)[len('obs_cov_'):])
    avg_file_dict_all[obs_cov] = file

#print(file_dict)


for cov in file_dict_all:
    file_dict = {obs_cov: file_dict_all[obs_cov] for obs_cov in [cov]}
    avg_file_dict = {obs_cov: avg_file_dict_all[obs_cov] for obs_cov in [cov]}

    rate = rc.RateCalc(file_dict[cov],obs_gap)
    rate.fit_exp(tail=0.7)
    rate.plot_line(plot_folder, tag='obs_cov_{:.2f}'.format(cov), obs_cov=cov)
