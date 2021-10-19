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
import Lorenz96_alt as lorenz
import config as cf
import filter as fl
# first 10 odd primes as random seeds
seeds = [3]#, 5, 7, 11, 13, 17, 19, 23, 29, 31]
ev_time = 50
tts = []
for i in  range(4):
    tts.append(np.genfromtxt('../models/trajectory_{}_500.csv'.format(i+1), delimiter=',')[:ev_time])

true_trajectory = tts[0]
model, _ = lorenz.get_model(x0=true_trajectory[0], size=ev_time, obs_gap=0.1)

for cfile in ['../exp_rate/configs/gap_2/1_pc_2000/config_3.json']:#glob.glob('configs/*/*.json'):
    """ 
    observed_path = model.observation.generate_path(true_trajectory)
    cc = cf.ConfigCollector(expr_name = 'test', folder = '.')
    bpf = fl.ParticleFilter(model=model, particle_count=500, folder=cc.res_path)
    bpf.update(observed_path, resampling_method='systematic_noisy', noise=0.5)
    if bpf.status == 'success':
        bpf.plot_trajectories(true_trajectory, coords_to_plot=[0, 1, 8, 9],\
                                    file_path=cc.res_path + '/trajectories.png', measurements=False)
        bpf.compute_error(true_trajectory)
        bpf.plot_error(semilogy=True, resampling=False)    
    """    
    cfile = cfile.replace('\\', '/')
    parts = cfile.split('/')
    print(cfile, parts)
    results_folder = '../results/gap_2/1_pc_2000/config_3_seed_3'#{}'.format(parts[1])

    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)



    index = int(parts[-3].split('_')[-1]) - 1
    batch_experiment = be.BPFBatchObs(cfile, tts[index], seeds, results_folder)
    batch_experiment.run()
#"""