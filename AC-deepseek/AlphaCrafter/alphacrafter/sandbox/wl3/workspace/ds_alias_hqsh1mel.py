import json, os
factors = {}
for f in sorted(os.listdir('factors')):
    if f.endswith('.json') and not f.endswith('.bak') and f != 'factor_ensemble.json':
        try:
            p = json.load(open(f'factors/{f}'))
            vid = p.get('factor_id')
            st = p.get('validation', {}).get('status')
            m = p.get('validation', {}).get('metrics', {})
            lv = p.get('validation', {}).get('last_validated')
            factors[vid] = (st, m.get('ic'), m.get('icir'), m.get('ic_recent_post_warmup'), m.get('icir_recent_post_warmup'), lv, m.get('max_abs_library_correlation'))
        except Exception as e:
            print('err', f, e)
for k, v in sorted(factors.items()):
    print(f"{k:32s} status={v[0]:10s} ic={v[1]:+.4f} icir={v[2]:+.3f} ic_recent={str(v[3]):>8s} icir_recent={str(v[4]):>8s} last_validated={v[5]} rho={v[6]}")
print()
print("count:", len(factors))