import glob
import os 
import json

gap_id = 5
obs_gap = 0.01

for folder in glob.glob('configs/gap_1/*'):
    folder = folder.replace('\\', '/')
    parts = folder.split('/')
    parts[1] = 'gap_{}'.format(gap_id) 
    new_folder =  '/'.join(parts)
    if not os.path.isdir(new_folder):
        os.makedirs(new_folder)
    for cfile in os.listdir(folder):
        with open('{}/{}'.format(folder, cfile), 'r') as cfl:
            data = json.load(cfl)
        data['obs_gap'] = obs_gap
        with open('{}/{}'.format(new_folder, cfile), 'w') as cfl:
            json.dump(data, cfl, indent=2)