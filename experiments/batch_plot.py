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
id_ = 1
dist_folder = 'dists/{}'.format(id_)
file_path = 'plots/{}_dist.png'.format(str(id_))

# compute distances
batch_plotter = be.AvgDistPlotter(dist_folder)
batch_plotter.plot(file_path, gap=4, ev_time=400, low_idx=0, high_idx=1)