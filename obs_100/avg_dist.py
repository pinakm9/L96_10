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

dist_folder = 'dists'
ev_time = [63]
num_seeds = 100
gap = 1

for i, obs_gap in enumerate([0.05]):
    w = np.zeros(ev_time[i])
    for file in glob.glob('{}/*obs_gap_{}*'.format(dist_folder, obs_gap)):
        w += pd.read_csv(file, delimiter=',')['sinkhorn_div'].to_numpy()
    print(w)
    data = {'time': [], 'sinkhorn_div': []}
    data['time'] += list(range(0, ev_time[i], gap))
    data['sinkhorn_div'] += list(w / num_seeds)
    df = pd.DataFrame(data)
    df.to_csv('avg_dists/obs_gap_{}.csv'.format(obs_gap), index=False)


