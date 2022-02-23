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

W = []
for obs_cov in [0.2, 0.4, 0.8, 1.6]:
    W.append(pd.read_csv('avg_dists/obs_cov_{}.csv'.format(obs_cov), delimiter=',')['sinkhorn_div'].to_numpy()) 

for i, obs_cov in enumerate([0.2, 0.4, 0.8, 1.6]):
    sample = W[i]
    rest = np.array(W[:i] + W[i+1:]).flatten()
    res = stats.ks_2samp(sample, rest)
    print(res.pvalue)