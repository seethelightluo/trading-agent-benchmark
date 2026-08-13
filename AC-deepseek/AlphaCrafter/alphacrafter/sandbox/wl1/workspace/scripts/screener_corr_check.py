import numpy as np

names = [l.strip() for l in open('scripts/_factor_corr_names.txt') if l.strip()]
M = np.load('scripts/_factor_corr_matrix.npy')
print("names:")
for i, n in enumerate(names):
    print(i, n)
print()
# map short ids to indices
short = {'rev_1d':'rev_1d','rev_2d':'rev_2d','rev_3d':'rev_3d','rev_5d':'rev_5d','nclv_1d':'nclv_1d','nclv_2d':'nclv_2d','nclv_3d':'nclv_3d','nclv_5d':'nclv_5d','id_rev_1d':'id_rev_1d','nbody_1d':'nbody_1d','rev_1d_vs':'rev_1d_vs','mom_120d_skip5':'mom_120d_skip5','vol_of_vol20x60':'vol_of_vol20x60','vix_beta_cond_60x20':'vix_beta_cond_60x20'}
idx = {}
for i, n in enumerate(names):
    for k in short:
        if n.endswith(k):
            idx[k] = i
print("idx:", idx)
active = ['rev_1d','rev_2d','rev_3d','rev_5d','nclv_1d','nclv_2d','nclv_3d','nclv_5d','id_rev_1d','nbody_1d','rev_1d_vs','mom_120d_skip5','vol_of_vol20x60','vix_beta_cond_60x20']
print("\nPairs |corr|>0.7 among active:")
for i in range(len(active)):
    for j in range(i+1, len(active)):
        a, b = active[i], active[j]
        if a in idx and b in idx:
            c = M[idx[a], idx[b]]
            if abs(c) > 0.7:
                print(f"  {a} ~ {b}: {c:.3f}")
