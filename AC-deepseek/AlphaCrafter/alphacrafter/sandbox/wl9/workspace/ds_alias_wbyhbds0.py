import json,glob,os
ids=['beta_VIX_60','kaufman_eff_20d','mom_120d_skip5','bb_width_20d','cny_beta_60','vol_z_20d','ac1_120d','mom_10d_skip5','dxy_corr_change_20_60','skew_20d']
for i in ids:
    try:
        d=json.load(open(f'factors/{i}.json'))
        v=d.get('validation',{})
        print(i,'|status=',v.get('status'),'|IC=',v.get('ic'),v.get('ic_mean'),'|ICIR=',v.get('icir'),d.get('expected_direction'))
    except Exception as e:
        print(i,'ERR',e)
print('=== check factor_ensemble.json location ===')
print(os.path.exists('factor_ensemble.json'))