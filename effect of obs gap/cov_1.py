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


priors = {'prior_1': [], 'prior_3': []}
obs_gaps = [0.01, 0.03, 0.05, 0.07, 0.09]
ev_times = {0.01: 315, 0.03: 105, 0.05:63, 0.07: 45, 0.09: 35}
files = {gap: {'prior_1': [], 'prior_3': []} for gap in obs_gaps}
obs_cov = 0.4


def get_cov(asml_file):
    h5 = tables.open_file(asml_file, mode='a')
    dim = np.array(getattr(h5.root.particles, 'time_' + str(0)).read().tolist()).shape[-1]
    ev_time = len(h5.root.observation.read().tolist())
    evals = np.zeros((1, dim))
    i = ev_time - 1
    ensemble = np.array(getattr(h5.root.particles, 'time_' + str(i)).read().tolist())
    evals[0, :] = scipy.linalg.eigh(np.cov(ensemble.T), eigvals_only=True)
    if hasattr(h5.root, 'eigenvalues'):
        h5.root.eigenvalues = evals
    else:
        h5.create_array(h5.root, 'eigenvalues', evals) 
    h5.close()
    return np.sum(evals, axis=-1), evals[:, -1] 



trace = {'obs gap': [], 'trace^-1': []}
heval = {'obs gap': [], 'eigenvalue^-1': []} 

for folder in glob.glob('data/*'):
    gap = float(re.search("\d+\.\d+", folder).group(0))
    prior = re.search("prior_\d", folder).group(0)
    #print(prior, gap) 
    if gap in obs_gaps:
        files[gap][prior].append(folder + '/assimilation.h5') 


j=-1

# plot bands
for prior in priors:
    fig_t = plt.figure(figsize=(8, 8))
    fig_h = plt.figure(figsize=(8, 8))
    ax_t = fig_t.add_subplot(111)
    ax_h = fig_h.add_subplot(111)

    for obs_gap in obs_gaps:
        t = [obs_gap]

        for asml_file in files[obs_gap][prior]:
            print('working on file {}'.format(asml_file), end='\r') 
            trace['obs gap'] += t 
            heval['obs gap'] += t

            tr, eval = get_cov(asml_file)
            #print(tr.shape, eval.shape, obs_gap, len(t))

            trace['trace^-1'] += [1./ tr[j]]
            heval['eigenvalue^-1'] += [1. / eval[j]]

    sns.lineplot(ax=ax_t, data=trace, x='obs gap', y='trace^-1', ci='sd')
    sns.lineplot(ax=ax_h, data=heval, x='obs gap', y='eigenvalue^-1', ci='sd')

    fig_t.savefig('plots/last_trace_inv_{}_obs_cov_{}.png'.format(prior, obs_cov))
    fig_h.savefig('plots/last_heval_inv_{}_obs_cov_{}.png'.format(prior, obs_cov))



