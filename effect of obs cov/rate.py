import sys
from pathlib import Path
from os.path import dirname, abspath
from cv2 import threshold
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
print(module_dir)

import numpy as np
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, re
from sklearn.linear_model import LinearRegression
import utility as ut

obs_gap = 0.05

class RateComputer:

    def __init__(self, file_dict, avg_file_dict) -> None:
        self.file_dict = file_dict
        self.avg_file_dict = avg_file_dict
        self.obs_covs = list(file_dict.keys())       


    def read(self, file, k=None):
        dist = pd.read_csv(file, delimiter=',')['sinkhorn_div'].to_numpy()[:k]
        if k is not None:
            dist = dist[:k]
        phy_time = obs_gap * np.arange(0., len(dist), 1.)
        return dist, phy_time
    
    
    def collect_data(self, k=None):
        dist_all = []
        phy_time_all = []
        for obs_cov in self.obs_covs:
            dist = []
            phy_time = []
            for i, file in enumerate(self.file_dict[obs_cov]):
                d, t = self.read(file, k)
                dist.append(d)
                phy_time.append(t)
            dist_all += list(np.array(dist).T)
            phy_time_all += list(np.array(phy_time).T)

        return np.array(dist_all).reshape(-1), np.array(phy_time_all).reshape(-1, 1) 

    #@ut.timer
    def fit_line(self, dist, phy_time):
        idx = np.where(dist > 0.)[0]
        y = np.log(dist[idx])
        phy_time = phy_time[idx]
        reg = LinearRegression().fit(phy_time, y)
        return reg.coef_, reg.intercept_, reg.score(phy_time, y)


    
    def find_tail_mean(self):
        for obs_cov, file in self.avg_file_dict.items():
            dist, _ = self.read(file)
            t = int(len(dist) * 0.7)
            self.tail_mean = np.mean(dist[t:]) 
            print('tail mean found at {}'.format(self.tail_mean))


    
    def find_stop_point(self):
        for obs_cov, file in self.avg_file_dict.items():
            dist, _ = self.read(file)
            dist = dist[:-1] - dist[1:]
            self.stop_point = np.where(dist < 0.)[0][0]
            print('stop point found at {}'.format(self.stop_point))


    
    @ut.timer
    def fit(self, k):
        self.find_stop_point()
        dist, phy_time = self.collect_data(self.stop_point)
        self.find_tail_mean()
        self.coef, self.intercept, self.score = self.fit_line(dist - self.tail_mean, phy_time.reshape(-1, 1))
    

    def plot_scores(self, folder):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.plot(range(2, len(self.scores)+2), self.scores)
        ax.set_ylabel('coefficient of determination')
        ax.set_xlabel('time points used')
        plt.savefig('{}/scores.png'.format(folder))

    
    def plot_line(self, folder):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        dist, phy_time = self.collect_data()
        phy_time = phy_time.flatten()
        ax.scatter(phy_time, dist, s=5, c='grey', alpha=0.2)
        m, c = self.coef[-1], self.intercept
        t = np.unique(phy_time)
        label = r'${:.2f}\,\exp({:.2f}t) + {:.2f}$'.format(np.exp(c), m, self.tail_mean)
        ax.plot(t, self.tail_mean + np.exp(m*t + c), c='deeppink', label=label)
        ax.set_ylabel('sinkhorn divergence')
        ax.set_xlabel(r'time ($t$)')
        ax.set_title('obs gap = {:.2f}, obs cov = {:.2f}'.format(obs_gap, self.obs_covs[0]))
        plt.legend()
        plt.savefig('{}/rate_obs_cov_{:.2f}.png'.format(folder, self.obs_covs[0]))


dist_folder = 'dists'
avg_dist_folder = 'avg_dists'
plot_folder = 'plots'


# collect files
file_dict_all = {obs_cov:[] for obs_cov in [0.2, 0.4, 0.8, 1.6]}
avg_file_dict_all = {}

for file in glob.glob('{}/*.csv'.format(dist_folder)):
    obs_cov = float(re.search("obs_cov_\d+\.\d+", file).group(0)[len('obs_cov_'):])
    file_dict_all[obs_cov].append(file)

for file in glob.glob('{}/*.csv'.format(avg_dist_folder)):
    obs_cov = float(re.search("obs_cov_\d+\.\d+", file).group(0)[len('obs_cov_'):])
    avg_file_dict_all[obs_cov] = file

#print(file_dict)


for cov in file_dict_all:
    file_dict = {obs_cov: file_dict_all[obs_cov] for obs_cov in [cov]}
    avg_file_dict = {obs_cov: avg_file_dict_all[obs_cov] for obs_cov in [cov]}

    rc = RateComputer(file_dict, avg_file_dict)
    rc.fit(200)
    rc.plot_line(plot_folder)