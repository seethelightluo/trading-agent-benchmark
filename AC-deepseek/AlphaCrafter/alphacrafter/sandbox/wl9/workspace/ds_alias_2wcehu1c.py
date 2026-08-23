import json
def met(f):
    d=json.load(open(f'factors/{f}.json'))
    m=d.get('validation',{}).get('metrics',{})
    return m
allf=['ac1_120d','bb_width_20d','beta_VIX_60','cny_beta_60','days_since_high_60','dxy_corr_change_20_60','kaufman_eff_20d','kurt_20d','mom_10_vixreg','mom_10d_skip5','mom_120d_skip5','rng_pos_20d','skew_20d','streak_len_14','vix_beta_cond_60x20','vix_roc_20d','vol_z_20d']
print(f"{'factor':26s} {'IC':>7} {'ICIR':>7} {'q':>9} {'turn':>6} {'dir':>4}")
for f in allf:
    m=met(f)
    ic=m.get('ic'); icir=m.get('icir')
    q=None if (ic is None or icir is None) else abs(ic)*abs(icir)
    print(f"{f:26s} {str(ic):>7} {str(icir):>7} {str(q if q is None else round(q,5)):>9} {str(m.get('turnover_10d_rank')):>6} {str(json.load(open(f'factors/{f}.json')).get('expected_direction')):>4}")