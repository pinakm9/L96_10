# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import numpy as np
from bpf_plotter import plot_ensemble_evol

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
plot_ensemble_evol(db_path, hidden_path, dims, time_idx, time_factor=1,\
					   hidden_color = 'red', prior_mean_color = 'purple', posterior_mean_color = 'maroon',\
					   obs_inv = None, obs_inv_color = 'black',\
					   fig_size = (10, 10), pt_size = 80, size_factor = 5,\
					   dpi = 300, ens_colors = ['orange', 'green'], alpha = 0.5, pdf_resolution = 300, attractor=attractor)