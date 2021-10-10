# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath

from tables import description
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import batch_expr as be 
import os, shutil
import numpy as np
import glob

# move folders if necessary
for folder in glob.glob('bpf_dists/*/*/'):
    if 'pc' in folder:
        folder = folder.replace('\\', '/')
        parts = folder.split('/')
        parts.insert(len(parts)-2, parts[2].split('_')[0])
        destination = '/'.join(parts)
        shutil.move(folder, destination)

fnames = ['1_vs_2', '1_vs_3', '2_vs_3']

pc_idx = [0, 1, 2, 3]
for folder in glob.glob('bpf_dists/*/*/'):
    dist_folder = folder.replace('\\', '/')
    parts = dist_folder.split('/')
    print('working on distance folder {}'.format(dist_folder))
    plot_folder = '/'.join(['plots'] + parts[1:])
    if plot_folder.endswith('/'):
        plot_folder = plot_folder[:-1]
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    for low_idx in [0, 1, 2]:
        batch_plotter = be.AvgDistPlotter(dist_folder)
        batch_plotter.plot(plot_folder, gap=1, ev_time=50, low_idx=low_idx, pc_idx=pc_idx, inset=False, y_lims=None)

"""
fnames = ['1_vs_2', '1_vs_3', '2_vs_3']
pc_idx = [0, 1, 2, 3]
for id_ in [0, 1]:
    for low_idx in [0, 1, 2]:
        high_idx = low_idx + 1
        dist_folder = 'dists/{}'.format(id_)

        file_path = 'plots/{}/dist_{}.png'.format(id_, fnames[low_idx])
        # plot distances
        batch_plotter = be.AvgDistPlotter(dist_folder, initial_dist_folder)
        batch_plotter.plot(file_path, gap=1, ev_time=400, low_idx=low_idx, high_idx=high_idx, pc_idx=pc_idx, inset=False, y_lims=[1.5,6.0])
"""