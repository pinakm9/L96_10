# add models folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
model_dir = str(script_dir.parent)
sys.path.insert(0, model_dir + '/models')

import numpy as np 
import json
import filter as fl
import config as cf
import Lorenz96_alt as model
import copy
import os
import tables
import tensorflow as tf
import wasserstein as ws
import utility as ut
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class BPFBatchObs:
    """
    Runs a filtering experiment for a batch of observation realizations 
    """
    def __init__(self, config_file, true_trajectory, seeds, results_folder):
        self.true_trajectory = true_trajectory
        self.config_id = os.path.basename(config_file).split('.')[0]
        with open(config_file) as f:
            self.config = json.load(f)
        self.seeds = seeds 
        self.model = model.get_model(x0=self.true_trajectory[0], size=len(true_trajectory),\
                                     prior_cov=self.config['prior_cov'],\
                                     obs_cov=self.config['obs_cov'], shift=self.config['shift'],\
                                     obs_gap=self.config['obs_gap'])[0]
        self.results_folder = results_folder
        self.config['assimilation_steps'] = len(true_trajectory)

    def run_single_expr(self, seed):
        # set random seed
        np.random.seed(seed)
        # set up logging
        config = copy.deepcopy(self.config)
        config['seed'] = seed
        expr_name = '{}_seed_{}'.format(self.config_id, seed)
        cc = cf.ConfigCollector(expr_name = expr_name, folder = self.results_folder)
        self.bpf = fl.ParticleFilter(self.model, particle_count = self.config['particle_count'], folder = cc.res_path)
        # generate observation
        observed_path = self.model.observation.generate_path(self.true_trajectory)
        # assimilate
        print("starting assimilation ... ")
        self.bpf.update(observed_path, method = 'mean', resampling_method=self.config['resampling_method'],\
                        threshold_factor=self.config['resampling_threshold'], noise=self.config['resampling_cov'])
        # document results
        if self.bpf.status == 'success':
            self.bpf.plot_trajectories(self.true_trajectory, coords_to_plot=[0, 1, 8, 9],\
                                       file_path=cc.res_path + '/trajectories.png', measurements=False)
            self.bpf.compute_error(self.true_trajectory)
            self.bpf.plot_error(semilogy=True, resampling=False)
            config['status'] = self.bpf.status
            cc.add_params(config)
            cc.write(mode='json')

    @ut.timer
    def run(self):
        for seed in self.seeds:
            self.run_single_expr(seed)


class BatchDist:
    """
    Computes distance for a batch of experiments
    """
    def __init__(self, config_folder, seeds, results_folder, dist_folder):
        self.configs = [f.split('.')[0] for f in os.listdir(config_folder)]
        self.seeds = seeds
        self.results_folder = results_folder
        self.dist_folder = dist_folder
    
    @ut.timer
    def run_for_pair(self, config_id_1, config_id_2, seed, epsilon=0.01, num_iters=200, p=2):
        # find the right folders
        for f in os.listdir(self.results_folder):
            if f.startswith(config_id_1) and f[:-2].endswith(str(seed)):
                folder_1 = self.results_folder + '/' + f
            if f.startswith(config_id_2) and f[:-2].endswith(str(seed)):
                folder_2 = self.results_folder + '/' + f
        # figure out number of assimilation steps
        with open(folder_1 + '/config.json') as f:
            ev_time = json.load(f)['assimilation_steps']
        
        file_1 = tables.open_file(folder_1 + '/assimilation.h5')
        file_2 = tables.open_file(folder_2 + '/assimilation.h5')

        dist = np.zeros(ev_time)
        for t in range(ev_time):
            print('computing distance for step #{}'.format(t))
            ensemble_1 = np.array(getattr(file_1.root.particles, 'time_' + str(t)).read().tolist())
            ensemble_2 = np.array(getattr(file_2.root.particles, 'time_' + str(t)).read().tolist())
            #weights_1 = np.array(getattr(file_1.root.weights, 'time_' + str(t)).read().tolist())
            #weights_2 = np.array(getattr(file_2.root.weights, 'time_' + str(t)).read().tolist())
            ensemble_1 = tf.convert_to_tensor(ensemble_1, dtype=tf.float32)
            ensemble_2 = tf.convert_to_tensor(ensemble_2, dtype=tf.float32)
            #weights_1 = tf.convert_to_tensor(weights_1, dtype=tf.float32)
            #weights_2 = tf.convert_to_tensor(weights_2, dtype=tf.float32)
            dist[t] = (ws.sinkhorn_div_tf(ensemble_1, ensemble_2,\
                       epsilon=epsilon, num_iters=num_iters, p=p).numpy())**(1./p)

        id_1 = config_id_1.split('_')[-1]
        id_2 = config_id_2.split('_')[-1]
        file_path = '{}/{}_vs_{}_seed_{}.npy'.format(self.dist_folder, id_1, id_2, seed)
        #np.save(file_path, dist)
        file_1.close()
        file_2.close()
        return dist

    @ut.timer
    def run(self, epsilon=0.01, num_iters=200, p=2):
        for j, config_id_1 in enumerate(self.configs):
            for config_id_2 in self.configs[j+1:]:
                data = {'time': [], 'seed':[], 'sinkhorn_div': []}
                for i, seed in enumerate(self.seeds):
                    print('comparing {} and {} for seed: {}'.format(config_id_1, config_id_2, seed))
                    dist = self.run_for_pair(config_id_1, config_id_2, seed, epsilon, num_iters, p)
                    ev_time = len(dist)
                    data['time'] += list(range(ev_time))
                    data['seed'] += [seed] * ev_time 
                    data['sinkhorn_div'] += list(dist)
                df = pd.DataFrame(data)
                id_1 = config_id_1.split('_')[-1]
                id_2 = config_id_2.split('_')[-1]
                df.to_csv('{}/{}_vs_{}.csv'.format(self.dist_folder, id_1, id_2), index=False)


class AvgDistPlotter:
    """
    Plots average distance for same number of particles
    """
    def __init__(self, dist_folder):
        self.dist_folder = dist_folder 
        self.particle_counts = {f: int(f.split('_')[-1]) for f in os.listdir(dist_folder)}
        self.max_particle_count = max(self.particle_counts.values())
        self.colors = ['red', 'green', 'blue', 'orange', 'grey', 'purple']

    def plot(self, file_path):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        for folder in os.listdir(self.dist_folder):
            for i, f in enumerate(os.listdir(self.dist_folder + '/' + folder)):
                df = pd.read_csv(self.dist_folder + '/' + folder + '/' + f)
                if self.particle_counts[folder] != self.max_particle_count:
                    sns.lineplot(data=df, x=df['time'], y=df['sinkhorn_div'], color=self.colors[i], ci='sd', ax=ax)
                else:
                    sns.lineplot(data=df, x=df['time'], y=df['sinkhorn_div'], color=self.colors[i], ci='sd', ax=ax)
                #plt.show()
                plt.savefig(file_path)



