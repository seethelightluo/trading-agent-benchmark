import json
print("=== factors/factor_ensemble.json ===")
print(open('factors/factor_ensemble.json').read())
print()
for f in ['vol_adj_mom_accel_20x60.json', 'dn_mkt_beta_60d.json', 'rate_beta_cn10y_60d.json']:
    print('='*20, f, '='*20)
    txt = open('factors/'+f).read()
    print(txt[:1500])
