import sys
from pathlib import Path
from os.path import dirname, abspath
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
import os
import re
import tables
import scipy


num_obs_gaps = 30
priors = {'prior_1': []}
obs_gaps = [0.01, 0.03, 0.05, 0.07, 0.09][:2]
ev_times = {gap : 10 for gap in obs_gaps}
files = {gap: {'prior_1': []} for gap in obs_gaps}
obs_cov = 0.4


def get_cov(asml_file):
    h5 = tables.open_file(asml_file, mode='a')
    dim = np.array(getattr(h5.root.particles, 'time_' + str(0)).read().tolist()).shape[-1]
    ev_time = len(h5.root.observation.read().tolist())

    eval = np.array(h5.root.predicted_evals.read().tolist())
    a_heval = eval[:, 0]
    a_trace = eval[:, 1]

    eval = np.array(h5.root.corrected_evals.read().tolist())
    b_heval = eval[:, 0]
    b_trace = eval[:, 1]

    eval = np.array(h5.root.resampled_evals.read().tolist())
    r_heval = eval[:, 0]
    r_trace = eval[:, 1] 
            
    return a_heval, a_trace, b_heval, b_trace, r_heval, r_trace 



trace = {'obs gap': [], 'trace': []}
heval = {'obs gap': [], 'eigenvalue': []} 

for folder in glob.glob('data/*'):
    gap = float(re.search("\d+\.\d+", folder).group(0))
    prior = re.search("prior_\d", folder).group(0)
    #print(prior, gap) 
    if gap in obs_gaps:
        files[gap][prior].append(folder + '/assimilation.h5') 


j=5

# plot bands
gaps = [0.01, 0.03]

fig_t = plt.figure(figsize=(8, 8))
fig_h = plt.figure(figsize=(8, 8))
fig_ht = plt.figure(figsize=(8, 8))
fig_hh = plt.figure(figsize=(8, 8))
ax_t = fig_t.add_subplot(111)
ax_h = fig_h.add_subplot(111)
ax_ht = fig_t.add_subplot(111)
ax_hh = fig_h.add_subplot(111)

a_heval, a_trace = [[], []], [[], []]
b_heval, b_trace = [[], []], [[], []]
r_heval, r_trace = [[], []], [[], []]
ha_heval, ha_trace = [], []
hb_heval, hb_trace = [], []
hr_heval, hr_trace = [], []
n_steps = [21, 8]
t = [gap*np.arange(0, n_step, 1.) for (gap, n_step) in zip(gaps, n_steps)]
a_markers = ['.', 'o']
b_markers = ['*', 'p']
r_markers = ['x', 'X']
colors = ['deeppink', 'grey']
alpha = [1.0, 0.5]

for i, obs_gap in enumerate(gaps):

    for asml_file in files[obs_gap][prior]:
        print('working on file {}'.format(asml_file), end='\r') 
       
        a_h, a_t, b_h, b_t, r_h, r_t = get_cov(asml_file)
        #print(a_h), a_t, r_h, r_t)
        a_heval[i].append(a_h[:n_steps[i]])
        a_trace[i].append(a_t[:n_steps[i]])
        b_heval[i].append(b_h[:n_steps[i]])
        b_trace[i].append(b_t[:n_steps[i]])
        r_heval[i].append(r_h[:n_steps[i]])
        r_trace[i].append(r_t[:n_steps[i]])
        """
        ha_heval[i] += list(a_h)
        ha_trace[i] += list(a_t)
        hb_heval[i] += list(b_h)
        hb_trace[i] += list(b_t)
        hr_heval[i] += list(r_h)
        hr_trace[i] += list(r_t)
        """
    a_heval[i] = np.sum(np.array(a_heval[i]), axis=0) / len(files[obs_gap]['prior_1'])
    a_trace[i] = np.sum(np.array(a_trace[i]), axis=0) / len(files[obs_gap]['prior_1'])
    b_heval[i] = np.sum(np.array(b_heval[i]), axis=0) / len(files[obs_gap]['prior_1'])
    b_trace[i] = np.sum(np.array(b_trace[i]), axis=0) / len(files[obs_gap]['prior_1'])
    r_heval[i] = np.sum(np.array(r_heval[i]), axis=0) / len(files[obs_gap]['prior_1'])
    r_trace[i] = np.sum(np.array(r_trace[i]), axis=0) / len(files[obs_gap]['prior_1'])


    ax_h.scatter(t[i], a_heval[i], marker=a_markers[i], c=colors[i], alpha=alpha[i], label='predicted, obs gap = {:.2f}'.format(obs_gap))
    ax_h.scatter(t[i], b_heval[i], marker=b_markers[i], c=colors[i], alpha=alpha[i], label='corrected, obs gap = {:.2f}'.format(obs_gap))
    ax_h.scatter(t[i], r_heval[i], marker=r_markers[i], c=colors[i], alpha=alpha[i], label='resampled, obs gap = {:.2f}'.format(obs_gap))
    ax_t.scatter(t[i], a_trace[i], marker=a_markers[i], c=colors[i], alpha=alpha[i], label='predicted, obs gap = {:.2f}'.format(obs_gap))
    ax_t.scatter(t[i], b_trace[i], marker=b_markers[i], c=colors[i], alpha=alpha[i], label='corrected, obs gap = {:.2f}'.format(obs_gap))
    ax_t.scatter(t[i], r_trace[i], marker=r_markers[i], c=colors[i], alpha=alpha[i], label='resampled, obs gap = {:.2f}'.format(obs_gap))
    
    


ax_h.set_ylabel('largest eigenvalue of posterior covariance')
ax_h.set_xlabel('time')
ax_h.set_title('obs gap {} vs obs gap {}'.format(*gaps))
ax_t.set_ylabel('trace  of posterior covariance')
ax_t.set_xlabel('time')
ax_t.set_title('obs gap {} vs obs gap {}'.format(*gaps))

ax_h.legend()
ax_t.legend()
fig_t.savefig('plots/trace_{}_obs_gap_{}_vs_obs_gap_{}.png'.format(prior, *gaps))
fig_h.savefig('plots/heval_{}_obs_gap_{}_vs_obs_gap_{}.png'.format(prior, *gaps))



