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
particle_count = 1000
id_ = 0
config_folder = '../configs/{}_pc_{}'.format(id_, particle_count)
results_folder = 'results/{}_pc_{}'.format(id_, particle_count)
dist_folder = 'dists/{0}/{0}_pc_{1}'.format(id_, particle_count)

# compute distances
batch_dist = be.BatchDist(config_folder, seeds, results_folder, dist_folder)
batch_dist.run(gap=4, ev_time=400, epsilon=0.01, num_iters=200, p=2)