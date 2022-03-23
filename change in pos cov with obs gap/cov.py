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
obs_gaps = [0.005 * (2*i+1) for i in range(num_obs_gaps)]#[10:]
ev_times = {gap : 10 for gap in obs_gaps}
files = {gap: {'prior_1': []} for gap in obs_gaps}
obs_cov = 0.4


def get_cov(asml_file):
    h5 = tables.open_file(asml_file, mode='a')
    dim = np.array(getattr(h5.root.particles, 'time_' + str(0)).read().tolist()).shape[-1]
    ev_time = len(h5.root.observation.read().tolist())
    evals = np.zeros((ev_time, dim))
    for i in range(ev_time):
            ensemble = np.array(getattr(h5.root.particles, 'time_' + str(i)).read().tolist())
            evals[i, :] = scipy.linalg.eigh(np.cov(ensemble.T), eigvals_only=True)
    if hasattr(h5.root, 'eigenvalues'):
        h5.root.eigenvalues = evals
    else:
        h5.create_array(h5.root, 'eigenvalues', evals) 
    h5.close()
    return np.sum(evals, axis=-1), evals[:, -1] 



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

            trace['trace'] += [tr[j]]
            heval['eigenvalue'] += [eval[j]]

    sns.lineplot(ax=ax_t, data=trace, x='obs gap', y='trace', ci='sd')
    sns.lineplot(ax=ax_h, data=heval, x='obs gap', y='eigenvalue', ci='sd')

    fig_t.savefig('plots/trace_{}_obs_cov_{}_step_{}.png'.format(prior, obs_cov, j))
    fig_h.savefig('plots/heval_{}_obs_cov_{}_step_{}.png'.format(prior, obs_cov, j))



