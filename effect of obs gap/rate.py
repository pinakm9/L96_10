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

"""
X = np.array([1., 2., 4.])
y = np.array([4, 6., 9.])

reg = LinearRegression().fit(X, y)
print('kjhkjh', reg.score(X, y))
"""

class RateComputer:

    def __init__(self, file_dict, avg_file_dict) -> None:
        self.file_dict = file_dict
        self.avg_file_dict = avg_file_dict
        self.obs_gaps = list(file_dict.keys())       


    def read(self, file, obs_gap, k=None):
        dist = pd.read_csv(file, delimiter=',')['sinkhorn_div'].to_numpy()[:k]
        if k is not None:
            dist = dist[:k]
        phy_time = obs_gap * np.arange(0., len(dist), 1.)
        return dist, phy_time
    
    
    def collect_data(self, k=None):
        dist_all = []
        phy_time_all = []
        for obs_gap in self.obs_gaps:
            dist = []
            phy_time = []
            for i, file in enumerate(self.file_dict[obs_gap]):
                d, t = self.read(file, obs_gap, k)
                dist.append(d)
                phy_time.append(t)
            dist_all += list(np.array(dist).T)
            phy_time_all += list(np.array(phy_time).T)

        return np.array(dist_all).reshape(-1), np.array(phy_time_all).reshape(-1, 1) 

    #@ut.timer
    def fit_line(self, dist, phy_time):
        y = dist
        print(y)
        reg = LinearRegression().fit(phy_time, y)
        return reg.coef_, reg.intercept_, reg.score(phy_time, y)


    
    def find_tail_mean(self):
        for obs_gap, file in self.avg_file_dict.items():
            dist, _ = self.read(file, obs_gap)
            t = int(len(dist) * 0.7)
            self.tail_mean = np.mean(dist[t:]) 
            print('tail mean found at {}'.format(self.tail_mean))


    
    def find_stop_point(self):
        for obs_gap, file in self.avg_file_dict.items():
            dist, _ = self.read(file, obs_gap)
            dist = dist[:-1] - dist[1:]
            self.stop_point = np.where(dist < 0.)[0][0]
            print('stop point found at {}'.format(self.stop_point))


    
    @ut.timer
    def fit(self, k):
        self.find_stop_point()
        dist, phy_time = self.collect_data(self.stop_point)
        self.find_tail_mean()
        self.coef, self.intercept, self.score = self.fit_line(np.log(dist - self.tail_mean), phy_time.reshape(-1, 1))
    

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
        print(dist, phy_time)
        ax.scatter(phy_time, dist, s=5, alpha=0.5)
        m, c = self.coef[-1], self.intercept
        t = np.unique(phy_time)
        ax.scatter(t, self.tail_mean + np.exp(m*t + c), c='deeppink', s=10)
        plt.savefig('{}/line.png'.format(folder))


dist_folder = 'dists'
avg_dist_folder = 'avg_dists'
plot_folder = 'plots'


# collect files
file_dict_all = {obs_gap:[] for obs_gap in [0.01, 0.03, 0.05, 0.07, 0.09]}
avg_file_dict_all = {}

for file in glob.glob('{}/*.csv'.format(dist_folder)):
    obs_gap = float(re.search("obs_gap_\d+\.\d+", file).group(0)[len('obs_gap_'):])
    file_dict_all[obs_gap].append(file)

for file in glob.glob('{}/*.csv'.format(avg_dist_folder)):
    obs_gap = float(re.search("obs_gap_\d+\.\d+", file).group(0)[len('obs_gap_'):])
    avg_file_dict_all[obs_gap] = file

#print(file_dict)

file_dict = {obs_gap: file_dict_all[obs_gap] for obs_gap in [0.01]}
avg_file_dict = {obs_gap: avg_file_dict_all[obs_gap] for obs_gap in [0.01]}

rc = RateComputer(file_dict, avg_file_dict)
rc.fit(200)
#rc.plot_scores(plot_folder)
rc.plot_line(plot_folder)



print(rc.stop_point)