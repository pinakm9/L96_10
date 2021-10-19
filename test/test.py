import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
print(module_dir)
# import rest of the modules
import numpy as np
import Lorenz96_alt as lorenz
import filter as fl
import config as cf

# import model
ev_time = 50
true_trajectory = np.genfromtxt('../models/trajectory_2_500.csv', delimiter=',')[:ev_time]
#print(true_trajectory)
model, _ = lorenz.get_model(x0=true_trajectory[0], size=ev_time, obs_gap=0.2)
observed_path = model.observation.generate_path(true_trajectory)
cc = cf.ConfigCollector(expr_name = 'test_1', folder = '.')
bpf = fl.ParticleFilter(model=model, particle_count=500, folder=cc.res_path)
bpf.update(observed_path, resampling_method='systematic_noisy', noise=0.5)
if bpf.status == 'success':
    bpf.plot_trajectories(true_trajectory, coords_to_plot=[0, 1, 8, 9],\
                                file_path=cc.res_path + '/trajectories.png', measurements=False)
    bpf.compute_error(true_trajectory)
    bpf.plot_error(semilogy=True, resampling=False)
    #config['status'] = bpf.status
    #cc.add_params(config)
    #cc.write(mode='json')
