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
import tables
import wasserstein as ws
import tensorflow as tf
import pandas as pd

# first 10 odd primes as random seeds
seeds = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
index_map = list(range(1, len(seeds)+1, 1))
index_map2 = [0, 2, 4]
particle_count = 2000
id_ = 0
config_folder = '../configs/{}_pc_{}'.format(id_, particle_count)
results_folder = 'results/{}_pc_{}'.format(id_, particle_count)
enkf_folder = 'enkf_results'
dist_folder = 'enkf_dists/{}_pc_{}'.format(id_, particle_count)
ev_time = 400
gap = 4


# compute distances
for j, config in enumerate(sorted(os.listdir(config_folder))):
    config = config[:-5]
    print('working with config {}'.format(config))
    data = {'time': [], 'seed':[], 'sinkhorn_div': []}
    for i, seed in enumerate(seeds):
        # find the bpf file
        folder = config + '_seed_{}#0'.format(seed)
        print(folder)
        bpf_file =  tables.open_file(results_folder + '/' + folder + '/assimilation.h5', 'r')
        # find the enkf file
        folder = 'ob{}'.format(index_map[i])
        for f in os.listdir(enkf_folder + '/' + folder):
            if int(f[5:6]) == index_map2[j]:
                folder = folder + '/' + f
                break 
        #print('folder = {}'.format(folder))
        for f in os.listdir(enkf_folder + '/' + folder):
            if f.endswith('a_ensemble.npy'):
                #print('f', f)
                enkf_file = np.load(enkf_folder + '/' + folder + '/' + f)
                print(folder + '/' + f)
                #exit()
                break
        
        dist = np.zeros(int(ev_time / gap))
        for k, t in enumerate(range(0, ev_time, gap)):
            print('computing distance for step #{}'.format(t), end='\r')
            ensemble_bpf = tf.convert_to_tensor(np.array(getattr(bpf_file.root.particles, 'time_' + str(t)).read().tolist()),\
                                                dtype=tf.float32)
            #print(ensemble_bpf.shape, enkf_file[t, :, :,].T.shape)
            ensemble_enkf = tf.convert_to_tensor(enkf_file[t, :, :,].T, dtype=tf.float32)
            dist[k] = np.sqrt(ws.sinkhorn_div_tf(ensemble_enkf, ensemble_bpf, epsilon=0.01, num_iters=200, p=2).numpy())
        bpf_file.close()
        data['sinkhorn_div'] += list(dist)
        data['time'] += list(range(0, ev_time, gap))
        data['seed'] += [seed] * len(dist)
    df = pd.DataFrame(data)
    df.to_csv(dist_folder + '/{}.csv'.format(config), index=False)