python -c "
import json,glob,os
for f in ['ac1_120d','bb_width_20d','beta_VIX_60','cny_beta_60','days_since_high_60','dxy_corr_change_20_60','kaufman_eff_20d','kurt_20d','mom_10_vixreg','mom_10d_skip5','mom_120d_skip5','rng_pos_20d','skew_20d','streak_len_14','vix_beta_cond_60x20','vix_roc_20d','vol_z_20d']:
    d=json.load(open('factors/'+f+'.json'))
    v=d.get('validation',{})
    m=v.get('metrics',{}) if isinstance(v,dict) else {}
    print(f, '| dir',d.get('expected_direction'),'| ic',round(m.get('ic',0),4),'| icir',round(m.get('icir',0),4),'| hit',m.get('ic_hit_ratio'),'| sharpe',m.get('sharpe'),'| turn',m.get('turnover'))
"