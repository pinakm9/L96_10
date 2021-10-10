import json
import glob

gap_ids = [1, 2, 3, 4]
gaps = [0.1, 0.2, 0.3, 0.4]

for i, gap in enumerate(gaps):
    folder = 'configs/gap_{}'.format(gap_ids[i])

    for cfile in glob.glob(folder + '/*/*.json'):
        with open(cfile, 'r') as config:
            data = json.load(config)
        
        data['obs_gap'] = gap
        with open(cfile, 'w') as config:
            json.dump(data, config)