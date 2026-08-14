import json, glob
for f in sorted(glob.glob('factors/*.json')):
    if 'bak' in f or 'signal' in f or 'ensemble' in f: continue
    try:
        d = json.load(open(f))
        v = d.get('validation', {})
        m = v.get('metrics', {})
        print(f"{d.get('factor_id','?'):35s} status={v.get('status','?'):12s} ic={m.get('ic', float('nan')):.4f} icir={m.get('icir', float('nan')):.3f} lv={v.get('last_validated','?')}")
    except Exception as e:
        print(f, 'ERR', e)