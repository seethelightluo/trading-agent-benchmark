import json,glob,os
ids=['beta_VIX_60','kaufman_eff_20d','mom_120d_skip5','bb_width_20d','cny_beta_60','vol_z_20d','ac1_120d','mom_10d_skip5','dxy_corr_change_20_60','skew_20d']
print("factor | ic | icir | hit | turnover")
for i in ids:
    d=json.load(open(f'factors/{i}.json'))
    m=d['validation']['metrics']
    print(f"{i:24s}| {m['ic']:+.4f} | {m['icir']:+.4f} | {m.get('ic_hit_ratio',0):.3f} | {m.get('turnover_10d_rank',0):.3f}")
# compute quality q=abs(ic)*abs(icir) for active set and verify weights
rows=[]
for i in ids:
    d=json.load(open(f'factors/{i}.json'))
    m=d['validation']['metrics']
    rows.append((i, abs(m['ic'])*abs(m['icir'])))
rows.sort(key=lambda x:-x[1])
total=sum(r for _,r in rows)
print('---q-tilt weights---')
for i,q in rows:
    print(f"{i:24s} q={q:.5f} w={q/total:.4f} dir={json.load(open(f'factors/{i}.json'))['expected_direction']}")