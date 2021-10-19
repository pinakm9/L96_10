import glob
import shutil, os
#"""
for stuff in glob.glob('bpf_dists/*/*/*/*'):
    if stuff.endswith('.csv'):
        os.remove(stuff)
#"""
"""
for folder in glob.glob('bpf_dists/*/*/*/*'):
    folder = folder.replace('\\', '/')
    new_folder = '/'.join(folder.split('/')[:-1])
    for fl in os.listdir(folder):
        shutil.move(folder + '/{}'.format(fl), new_folder + '/{}'.format(fl))
"""