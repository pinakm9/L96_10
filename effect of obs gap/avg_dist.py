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
ev_time = 100
num_seeds = 10
gap = 1

for obs_gap in [0.01, 0.03, 0.05, 0.07, 0.09]:
    w = np.zeros(ev_time)
    for file in glob.glob('{}/*obs_gap_{}*'.format(dist_folder, obs_gap)):
        w += pd.read_csv(file, delimiter=',')['sinkhorn_div'].to_numpy()
    print(w)
    data = {'time': [], 'sinkhorn_div': []}
    data['time'] += list(range(0, ev_time, gap))
    data['sinkhorn_div'] += list(w / num_seeds)
    df = pd.DataFrame(data)
    df.to_csv('avg_dists/obs_cov_{}.csv'.format(obs_gap), index=False)


