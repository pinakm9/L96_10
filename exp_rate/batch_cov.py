import sys
from pathlib import Path
from os.path import dirname, realpath

from tables import description
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import glob
import os 
import tables
import batch_expr as be

for config_folder in glob.glob('configs/*/*'):
    config_folder = config_folder.replace('\\', '/')
    parts = config_folder.split('/')
    #print(parts)
    results_folder = 'results/{}/{}'.format(parts[1], parts[2])
    cov_folder = 'cov/{}/{}'.format(parts[1], parts[2])
    save_path = '{}/cov.png'.format(cov_folder)
    #print(save_path)
    if not os.path.isdir(cov_folder):
        os.makedirs(cov_folder)
    # compute cov
    batch_cov = be.BatchCov(config_folder, results_folder, cov_folder)
    batch_cov.run(gap=1, ev_time=200 if 'gap_5' in config_folder else 50)
    batch_cov.plot(save_path)
    