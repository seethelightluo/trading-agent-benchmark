import json, glob, re
def deep_keys(d, prefix="", depth=0):
    if depth>3: return
    if isinstance(d, dict):
        for k,v in d.items():
            if k in ('metrics','ic','icir','ic_hit_ratio','sharpe','q','quality','ic_mean','ic_std','t_stat'):
                print(f"  {prefix}{k}: {v}")
            else:
                deep_keys(v, prefix+k+".", depth+1)
    elif isinstance(d,list):
        for i,v in enumerate(d[:2]):
            deep_keys(v,prefix+f"[{i}].",depth+1)
for f in ['beta_VIX_60','kaufman_eff_20d','mom_120d_skip5','bb_width_20d','cny_beta_60','vol_z_20d','ac1_120d','mom_10d_skip5','dxy_corr_change_20_60','skew_20d']:
    d=json.load(open(f'factors/{f}.json'))
    print("===",f,"dir",d.get('expected_direction'))
    deep_keys(d)