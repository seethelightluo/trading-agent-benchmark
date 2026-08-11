with open('scripts/_lib_signal_names.txt') as f:
    names = [l.strip() for l in f if l.strip()]
print("lib names:", names)
import numpy as np
M = np.load('scripts/_lib_signal_matrix.npy')
print("lib matrix shape:", M.shape)
# show NaN fraction per factor
for i, n in enumerate(names):
    print(f"  {n}: nan_frac={np.isnan(M[i]).mean():.3f}")