import json,glob,os
for f in ['beta_ew_60d','corr_ew_60','downside_vol_ratio_20','dxy_beta_cond_60x20','eurusd_beta_cond_60x20','kurt_20d_skip5','max_ret_20d','rel_mom_20d_skip5']:
    p=f'factors/{f}.json'
    if os.path.exists(p):
        d=json.load(open(p))
        print('===',f,'===')
        print(json.dumps(d,indent=1)[:800])
    print()