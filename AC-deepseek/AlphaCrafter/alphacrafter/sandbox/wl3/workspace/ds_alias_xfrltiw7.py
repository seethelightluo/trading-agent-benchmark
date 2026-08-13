import json, os
# inspect one library artifact shape + grid for comparison
import numpy as np
arr = np.load('factors/down_beta_60_signal.npy')
print("down_beta_60 artifact:", arr.shape)
arr2 = np.load('factors/eurusd_beta_cond_60x20_signal.npy')
print("eurusd artifact:", arr2.shape)
# list all artifact files
arts = sorted(f for f in os.listdir('factors') if f.endswith('_signal.npy'))
print(len(arts), "artifacts")