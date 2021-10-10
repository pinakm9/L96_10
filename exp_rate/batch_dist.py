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
import glob
# first 10 odd primes as random seeds
seeds = [3]#, 5, 7, 11, 13, 17, 19, 23, 29, 31]
"""
for id_ in [0, 1]:
    for pc in [250, 500, 1000, 2000]:
        particle_count = pc
        #id_ = 0
        config_folder = '../configs/{}_pc_{}'.format(id_, particle_count)
        results_folder = 'results/{}_pc_{}'.format(id_, particle_count)
        dist_folder = 'initial_dists/{0}_pc_{1}'.format(id_, particle_count)
"""
for cfolder in glob.glob('configs/*/*/'):
        # compute distances
        cfolder = cfolder.replace('\\', '/')
        print(cfolder)
        parts = cfolder.split('/')
        results_folder = 'results/{}/{}'.format(parts[1], parts[2])
        print(results_folder)
        dist_folder = 'bpf_dists/{}/{}'.format(parts[1], parts[2])
        if not os.path.isdir(dist_folder):
            os.makedirs(dist_folder)
        batch_dist = be.BatchDist(cfolder, seeds, results_folder, dist_folder)
        batch_dist.run(gap=1, ev_time=None, epsilon=0.01, num_iters=200, p=2)