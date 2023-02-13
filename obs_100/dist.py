import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
print(module_dir)

import wasserstein as ws
import numpy as  np
import Lorenz96_alt as lorenz
import filter as fl
import matplotlib.pyplot as plt
import json
import pandas as pd
import os

seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]#, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541]
num_seeds = len(seeds)
num_obs_gaps = 1
num_experiments = num_seeds * num_obs_gaps
x0 = np.genfromtxt('../models/trajectory_1_500.csv', dtype=np.float64, delimiter=',')[-1]

model_params = {}
model_params['x0'] = [x0] * num_experiments
model_params['ev_time'] = [int(i) for i in np.repeat([63], num_seeds)]
model_params['prior_cov'] = [0.1] * num_experiments
model_params['shift'] = [0.0] * num_experiments
model_params['obs_gap'] = np.repeat([0.05], num_seeds)
model_params['obs_cov'] = [0.4 for i in range(num_experiments)]

experiment_params = {}
experiment_params['num_asml_steps'] = model_params['ev_time']
experiment_params['obs_seed'] = seeds * num_obs_gaps
experiment_params['filter_seed'] = [3] * num_experiments
experiment_params['coords_to_plot'] = [[0, 1, 8, 9]] * num_experiments
experiment_params['tag'] = ['prior_1_obs_gap_{}_obs_cov_{}_seed_{}'.format(gap, cov, seed) \
                            for gap, cov, seed in zip(model_params['obs_gap'], model_params['obs_cov'], experiment_params['obs_seed'])] 
 
_model_params = {}
_model_params['x0'] = [x0] * num_experiments
_model_params['ev_time'] = [int(i) for i in np.repeat([63], num_seeds)]
_model_params['prior_cov'] = [1.0] * num_experiments
_model_params['shift'] = [4.0] * num_experiments
_model_params['obs_gap'] = np.repeat([0.05], num_seeds)
_model_params['obs_cov'] = [0.4 for i in range(num_experiments)]

_experiment_params = {}
_experiment_params['num_asml_steps'] = model_params['ev_time']
_experiment_params['obs_seed'] = seeds * num_obs_gaps
_experiment_params['filter_seed'] = [3] * num_experiments
_experiment_params['coords_to_plot'] = [[0, 1, 8, 9]] * num_experiments
_experiment_params['tag'] = ['prior_3_obs_gap_{}_obs_cov_{}_seed_{}'.format(gap, cov, seed) \
                            for gap, cov, seed in zip(model_params['obs_gap'], model_params['obs_cov'], experiment_params['obs_seed'])] 


filter_params = {}
filter_params['particle_count'] = [500] * num_experiments
filter_params['threshold_factor'] = [1.0] * num_experiments 
filter_params['resampling_method']  = ['systematic_noisy'] * num_experiments
filter_params['resampling_cov'] = [0.5] * num_experiments

batch_experiment = fl.BatchExperiment(get_model_funcs=[lorenz.get_model] * num_experiments, model_params=model_params, experiment_params=experiment_params,\
                            filter_types=[fl.ParticleFilter] * num_experiments, filter_params=filter_params, folders=['data'] * num_experiments)
_batch_experiment = fl.BatchExperiment(get_model_funcs=[lorenz.get_model] * num_experiments, model_params=_model_params, experiment_params=_experiment_params,\
                            filter_types=[fl.ParticleFilter] * num_experiments, filter_params=filter_params, folders=['data'] * num_experiments)

folder_list_1 = [experiment.folder for experiment in batch_experiment.get_exps()]
folder_list_2 = [experiment.folder for experiment in _batch_experiment.get_exps()]

#print(folder_list_2)

dist_folder = 'dists/5'
batch_dist = ws.BatchDist(folder_list_1, folder_list_2, dist_folder)
batch_dist.run(gap=1, ev_time=None, epsilon=0.5)

