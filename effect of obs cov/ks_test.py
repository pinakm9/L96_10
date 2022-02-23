import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
print(module_dir)

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from ks2d import ks2d2s






W, obs_gap = [], 0.05
for obs_cov in [0.2, 0.4, 0.8, 1.6]:
    W.append(pd.read_csv('avg_dists/obs_cov_{}.csv'.format(obs_cov), delimiter=',')['sinkhorn_div'].to_numpy()) 


def plot(index):
    sample = np.array(W[index])
    t = np.array([obs_gap*i for i in range(len(sample))])
    rest, T = [], []
    for i in range(len(W)):
        if i != index:
            rest += [W[index]]
            T += [t]
    rest = np.array(rest).flatten()
    T = np.array(T).flatten() 

    p, ks = ks2d2s(sample, t, rest, T, extra=True)
    print(p, ks)


for i, obs_cov in enumerate([0.2, 0.4, 0.8, 1.6]):
    plot(i)