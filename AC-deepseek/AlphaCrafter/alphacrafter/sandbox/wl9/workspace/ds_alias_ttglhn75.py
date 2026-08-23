import json, glob
def met(f):
    d=json.load(open(f'factors/{f}.json'))
    m=d.get('validation',{}).get('metrics',{})
    q=abs(m.get('ic',0))*abs(m.get('icir',0))
    return m.get('ic',0), m.get('icir',0), q, m.get('turnover_10d_rank'), d.get('expected_direction')
allf=['ac1_120d','bb_width_20d','beta_VIX_60','cny_beta_60','days_since_high_60','dxy_corr_change_20_60','kaufman_eff_20d','kurt_20d','mom_10_vixreg','mom_10d_skip5','mom_120d_skip5','rng_pos_20d','skew_20d','streak_len_14','vix_beta_cond_60x20','vix_roc_20d','vol_z_20d']
rows=[]
for f in allf:
    ic,icir,q,turn,ed=met(f)
    rows.append((f,ic,icir,q,turn,ed))
print(f"{'factor':26s} {'IC':>7} {'ICIR':>7} {'q':>8} {'turn':>6} {'dir':>4}")
for r in sorted(rows,key=lambda x:-x[3]):
    print(f"{r[0]:26s} {r[1]:7.4f} {r[2]:7.4f} {r[3]:8.5f} {r[4]:6.2f} {r[5]:>4}")