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
ev_time = 500
tts = []
for i in  range(5):
    tts.append(np.genfromtxt('../models/trajectory_{}_500.csv'.format(i+1), delimiter=',', dtype=np.float64)[:ev_time])


for cfile in glob.glob('configs/*/*/*.json'):
    if 'gap_5' in cfile:
        cfile = cfile.replace('\\', '/')
        parts = cfile.split('/')
        print(cfile, parts)
        results_folder = 'results/{}/{}'.format(parts[1], parts[2])
        if not os.path.isdir(results_folder):
            os.makedirs(results_folder)
        index = int(parts[1].split('_')[-1]) - 1
        batch_experiment = be.BPFBatchObs(cfile, tts[index], seeds, results_folder)
        batch_experiment.run()
    

"""
id_ = 0
config_folder = '../configs/{}_pc_{}'.format(id_, particle_count)
particle_count = 500

results_folder = 'results/{}_pc_{}'.format(id_, particle_count)
for config_file in os.listdir(config_folder):
"""