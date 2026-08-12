import json, os
print("=== JSON files in factors/ (non-bak) ===")
for f in sorted(os.listdir('factors/')):
    if f.endswith('.json') and not f.endswith('.bak'):
        p = os.path.join('factors', f)
        try:
            d = json.load(open(p))
            v = d.get('validation', {})
            m = v.get('metrics', {})
            print(f"{f:45s} id={d.get('factor_id'):30s} status={v.get('status'):12s} ic={m.get('ic')} icir={m.get('icir')} last_validated={v.get('last_validated')}")
        except Exception as e:
            print(f, 'ERR', str(e)[:60])
print()
print("=== signal artifact sizes (provenance) ===")
for f in ['dn_mkt_beta_60d.json','rate_beta_cn10y_60d.json','vol_adj_mom_accel_20x60.json']:
    d = json.load(open('factors/'+f))
    sa = d.get('validation',{}).get('signal_artifact',{})
    print(f, '->', sa.get('format'), sa.get('shape'), 'sha:', sa.get('sha256'))