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

fnames = ['1_vs_2', '1_vs_3', '2_vs_3']
pc_idx = [0, 1, 2, 3]
for id_ in [0, 1]:
    for low_idx in [0, 1, 2]:
        high_idx = low_idx + 1
        dist_folder = 'dists/{}'.format(id_)
        file_path = 'plots/{}/dist_{}.png'.format(id_, fnames[low_idx])
        # plot distances
        batch_plotter = be.AvgDistPlotter(dist_folder)
        batch_plotter.plot(file_path, gap=4, ev_time=400, low_idx=low_idx, high_idx=high_idx, pc_idx=pc_idx)