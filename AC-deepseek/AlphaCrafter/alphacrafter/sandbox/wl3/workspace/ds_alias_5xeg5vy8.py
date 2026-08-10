import json
for f in ['factors/bollinger_z_20d.json', 'factors/quarantine/mom_10d_skip5.json']:
    try:
        d = json.load(open(f))
        v = d.get('validation', {})
        print(f, '-> status:', v.get('status'), '| keys:', sorted(d.keys()))
        print('   metrics:', {k: v.get('metrics', {}).get(k) for k in ['ic','icir','ic_hit_ratio','coverage_asset_days','n_ic_dates','turnover_10d_rank','max_abs_library_correlation']})
        print('   artifact keys:', [k for k in v.keys() if 'artifact' in k or 'signal' in k])
    except Exception as e:
        print(f, 'ERR', e)