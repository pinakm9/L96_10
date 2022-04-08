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

obs_cov = 0.4



dist_folder = 'dists'
avg_dist_folder = 'avg_dists'
plot_folder = 'plots'


# collect files
file_dict_all = {obs_gap:[] for obs_gap in [0.01, 0.03, 0.05, 0.07, 0.09]}

for file in glob.glob('{}/*.csv'.format(dist_folder)):
    obs_gap = float(re.search("obs_gap_\d+\.\d+", file).group(0)[len('obs_gap_'):])
    file_dict_all[obs_gap].append(file)


for gap in file_dict_all:
    rate = rc.RateCalc(file_dict_all[gap], gap)
    rate.fit_exp(tail=0.9)
    rate.plot_line(plot_folder, tag='obs_gap_{:.2f}'.format(gap), obs_cov=obs_cov, ylim=(0.5, 12.0))
