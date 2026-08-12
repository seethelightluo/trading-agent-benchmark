import json, os
# Check effective factor validation details
for f in ['vol_adj_mom_accel_20x60','dn_mkt_beta_60d','rate_beta_cn10y_60d']:
    d = json.load(open(f'factors/{f}.json'))
    v = d['validation']
    print('='*60)
    print(f['factor_id'], '| status:', v.get('status'), '| last_validated:', v.get('last_validated'), '| period:', v.get('period'))
    m = v.get('metrics', {})
    print('  metrics keys:', list(m.keys()) if isinstance(m, dict) else type(m))
    if isinstance(m, dict):
        for k in ['ic','icir','coverage','turnover','decay','max_abs_library_correlation']:
            if k in m: print('   ', k, '=', m[k])
