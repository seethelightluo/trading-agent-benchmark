import json,glob
ids=['beta_VIX_60','kaufman_eff_20d','mom_120d_skip5','bb_width_20d','cny_beta_60','vol_z_20d','ac1_120d','mom_10d_skip5','dxy_corr_change_20_60','skew_20d']
for i in ids:
    try:
        d=json.load(open(f'factors/{i}.json'))
        keys={k:d[k] for k in d if k.lower() in ('ic','icir','sharpe','direction','name','category','ic_mean','hit_rate','quality','source','description','type','factor_type')}
        print(i, json.dumps(keys)[:200])
    except Exception as e:
        print(i,'ERR',e)
print('---ALL FILES---')
import os
for f in sorted(glob.glob('factors/*.json')):
    if '.bak' not in f and 'ensemble' not in f:
        print(os.path.basename(f))
