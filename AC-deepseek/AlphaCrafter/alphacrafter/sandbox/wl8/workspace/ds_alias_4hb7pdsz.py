import json, glob
for f in ['factors/evicted/mom_10d_skip5.json','factors/evicted/vix_beta_cond_60x20.json','factors/evicted/yield_beta_cond_60x20.json']:
    try:
        d=json.load(open(f))
        v=d.get('validation',{})
        sa=v.get('signal_artifact',{})
        print(f)
        print('  period:', v.get('period'), 'last_validated:', v.get('last_validated'), 'status:', v.get('status'))
        print('  artifact keys:', list(sa.keys()) if isinstance(sa,dict) else sa)
        print('  shape:', sa.get('shape') if isinstance(sa,dict) else None)
        print('  metrics:', {k:v.get('metrics',{}).get(k) for k in ['ic','icir','coverage_asset_days']})
    except Exception as e:
        print(f,'ERR',e)