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

np.random.seed(42)
X = np.repeat([0.5, 1.0, 2.0, 4.0], 10).reshape(-1, 1)
y = np.repeat(2.0 * np.array([0.5, 1.0, 2.0, 4.0]), 10) + np.random.normal(size=40)
reg = LinearRegression().fit(X, y)
print(reg.coef_)


np.random.seed(42)
X = np.array([0.5, 1.0, 2.0, 4.0]).reshape(-1, 1)
y = (np.repeat(2.0*np.array([0.5, 1.0, 2.0, 4.0]), 10) + np.random.normal(size=40)).reshape(-1, 10)
y = np.mean(y, axis=1).reshape(-1)
reg = LinearRegression().fit(X, y)
print(reg.coef_)