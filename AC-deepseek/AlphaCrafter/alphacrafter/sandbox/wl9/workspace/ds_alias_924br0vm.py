import json, glob
for f in sorted(glob.glob('factors/*.json')):
    if 'bak' in f or 'ensemble' in f or 'evicted' in f: continue
    try:
        d = json.load(open(f))
        v = d.get('validation', {})
        m = v.get('metrics', {})
        print(f"{d.get('factor_id','?'):22s} status={v.get('status','?'):10s} last_val={v.get('last_validated','?'):12s} period={v.get('period','?')[:25]:27s} ic={m.get('ic')} icir={m.get('icir')} horizon={v.get('admission_horizon', d.get('parameters',{}).get('horizon','?'))}")
    except Exception as e:
        print(f, 'ERR', e)