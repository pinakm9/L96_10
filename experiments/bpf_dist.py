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
particle_count_0 = 500
particle_count_1 = 1000
id_ = 1
config_folder = '../configs/{}_pc_{}'.format(id_, particle_count_0)
results_folder_0 = 'results/{}_pc_{}'.format(id_, particle_count_0)
results_folder_1 = 'results/{}_pc_{}'.format(id_, particle_count_1)

dist_folder = 'bpf_dists/{}'.format(id_)
ev_time = 400
gap = 4


# compute distances
for j, config in enumerate(sorted(os.listdir(config_folder))):
    config = config[:-5]
    print('working with {}'.format(config))
    data = {'time': [], 'seed':[], 'sinkhorn_div': []}
    for i, seed in enumerate(seeds):
        # find the bpf files
        folder = config + '_seed_{}#0'.format(seed)
        print(folder)
        bpf_file_0 =  tables.open_file(results_folder_0 + '/' + folder + '/assimilation.h5', 'r')
        bpf_file_1 =  tables.open_file(results_folder_1 + '/' + folder + '/assimilation.h5', 'r')
        
        dist = np.zeros(int(ev_time / gap))
        for k, t in enumerate(range(0, ev_time, gap)):
            print('computing distance for step #{}'.format(t), end='\r')
            ensemble_0 = tf.convert_to_tensor(np.array(getattr(bpf_file_0.root.particles, 'time_' + str(t)).read().tolist()),\
                                                dtype=tf.float32)
            ensemble_1 = tf.convert_to_tensor(np.array(getattr(bpf_file_1.root.particles, 'time_' + str(t)).read().tolist()),\
                                                dtype=tf.float32)
            dist[k] = np.sqrt(ws.sinkhorn_div_tf(ensemble_0, ensemble_1, epsilon=0.01, num_iters=200, p=2).numpy())
        bpf_file_0.close()
        bpf_file_1.close()
        data['sinkhorn_div'] += list(dist)
        data['time'] += list(range(0, ev_time, gap))
        data['seed'] += [seed] * len(dist)
    df = pd.DataFrame(data)
    df.to_csv(dist_folder + '/{}_{}_vs_{}.csv'.format(config, particle_count_0, particle_count_1), index=False)