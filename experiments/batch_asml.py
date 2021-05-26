# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import batch_expr as be 
import os
import numpy as np
# first 10 odd primes as random seeds
seeds = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
particle_count = 2000
ev_time = 500
id_ = 0
config_folder = '../configs/{}_pc_{}'.format(id_, particle_count)
true_trajectory = np.genfromtxt('../models/trajectory_500.csv', delimiter=',', dtype=np.float64)[:ev_time]
results_folder = 'results/{}_pc_{}'.format(id_, particle_count)
for config_file in os.listdir(config_folder):
    batch_experiment = be.BPFBatchObs(config_folder + '/' + config_file, true_trajectory, seeds, results_folder)
    batch_experiment.run()