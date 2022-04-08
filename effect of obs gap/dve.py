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
import dvl2

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
errro_dict_all = {obs_gap:{'prior_1': [], 'prior_3': []} for obs_gap in [0.01, 0.03, 0.05, 0.07, 0.09]}

for folder in glob.glob('{}/*'.format(error_folder)):
    file = folder + '/assimilation.h5'
    obs_gap = float(re.search("obs_gap_\d+\.\d+", file).group(0)[len('obs_gap_'):])
    prior = re.search("prior_\d", file).group(0)
    errro_dict_all[obs_gap][prior].append(file)


for gap in dist_dict_all:
    dvl = dvl2.DvL2(dist_dict_all[gap], errro_dict_all[gap]['prior_3'], gap)
    dvl.collect_mean_data_d()
    dvl.collect_mean_data_e()
    dvl.fit_line()
    dvl.plot_line(plot_folder, tag='obs_gap_{:.2f}'.format(gap), obs_cov=obs_cov)
