# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import batch_expr as be 
import numpy as np
# first 10 odd primes as random seeds
id_ = 1
particle_count = 2000
config_folder = '../configs/{}_pc_{}'.format(id_, particle_count)
results_folder = 'results/{}_pc_{}'.format(id_, particle_count)
cov_folder = 'cov/{}_pc_{}'.format(id_, particle_count)
save_path = 'plots/{}_pc_{}_cov.png'.format(id_, particle_count)

# compute distances
batch_cov = be.BatchCov(config_folder, results_folder, cov_folder)
batch_cov.run(gap=4, ev_time=400)
batch_cov.plot(save_path)