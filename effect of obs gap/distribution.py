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
import distr
import scipy.stats as stats

obs_cov = 0.4
dim = 5


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
    particles_1 = distr.Distribution(file_dict_all[gap]['prior_1'], gap)
    particles_1.collect_mean_data()
 
    particles_3 = distr.Distribution(file_dict_all[gap]['prior_3'], gap)
    particles_3.collect_mean_data() 


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    fig_dummy = plt.figure(figsize=(8, 8))
    ax_dummy = fig_dummy.add_subplot(111)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    
    data = particles_1.particles[:, dim]
    density = stats.gaussian_kde(data)
    n, x, _ = ax_dummy.hist(data, bins=np.linspace(0, 10., 50), histtype=u'step', density=True)  
    ax.plot(x, density(x), label=r'prior 1')

    data = particles_3.particles[:, dim]
    density = stats.gaussian_kde(data)
    n, x, _ = ax_dummy.hist(data, bins=np.linspace(-2, 10., 50), histtype=u'step', density=True)  
    ax.plot(x, density(x), label=r'prior 3')

    ax.set_ylabel(r'mean distribution of $x_{}$'.format(dim), fontsize=20)
    ax.set_title(r'$g = {:.2f}, \sigma = {:.2f}, t = {:.2f}$'.format(particles_1.obs_gap, obs_cov, particles_1.steps * gap), fontsize=20)
    fig.legend(fontsize=20)
    fig.savefig('{}/distribution_obs_gap_{:.2f}.png'.format(plot_folder, gap))
    