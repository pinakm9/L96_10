# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import numpy as np
import reduction as red

id_ = 1
particle_count = 250
config = 3 
seed = 3
ev_time = 400
db_path = 'results/{}_pc_{}/config_{}_seed_{}#0/assimilation.h5'.format(id_, particle_count, config, seed)
hidden_path = np.genfromtxt('../models/trajectory_500.csv', delimiter=',', dtype=np.float64)[:ev_time]
attractor = np.genfromtxt('../models/attractor_10000.csv', delimiter=',', dtype=np.float64)[:particle_count]
print(attractor.shape)
dims = [0, 1]
time_idx = list(range(100)) #+ list(range(20, ev_time, 4))
red.make_scree_plots(db_path, time_idx)