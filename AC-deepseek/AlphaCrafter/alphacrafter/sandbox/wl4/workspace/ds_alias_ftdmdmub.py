import json, glob
for f in ['factors/vol_adj_mom_accel_20x60.json','factors/dn_mkt_beta_60d.json','factors/rate_beta_cn10y_60d.json']:
    d=json.load(open(f))
    print('===', f)
    print('id:', d.get('factor_id'), '| status:', d.get('validation',{}).get('status'), '| last_validated:', d.get('last_validated'))
    print('metrics:', json.dumps(d.get('validation',{}).get('metrics',{}))[:600])
    print('keys:', list(d.keys()))
    print()