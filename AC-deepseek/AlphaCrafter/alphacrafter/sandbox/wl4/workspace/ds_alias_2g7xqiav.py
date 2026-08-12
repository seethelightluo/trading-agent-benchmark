import json
for fid in ['vol_adj_mom_accel_20x60','dn_mkt_beta_60d','rate_beta_cn10y_60d']:
    d = json.load(open(f'factors/{fid}.json'))
    v = d['validation']
    print('='*60)
    print(fid, '| status:', v.get('status'), '| last_validated:', v.get('last_validated'), '| period:', v.get('period'))
    m = v.get('metrics', {})
    print('  metrics:', json.dumps(m)[:800])
    print('  admission:', json.dumps(d.get('benchmark_admission',{}).get('selected_metrics',{}))[:400])
