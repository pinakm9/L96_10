import glob, os

for folder in glob.glob('results/*/*/'):
    folder = folder.replace('\\', '/')
    parts = folder.split('/')
    parts[2] = parts[2][:-2]
    new_folder = '/'.join(parts)
    os.rename(folder, new_folder)