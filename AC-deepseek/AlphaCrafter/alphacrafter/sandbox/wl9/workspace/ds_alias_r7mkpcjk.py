import json, glob, os
files = sorted(glob.glob('factors/*.json'))
for f in files:
    if 'ensemble' in f or f.endswith('.bak') or 'evicted' in f: continue
    d = json.load(open(f))
    v = d.get('validation',{})
    m = v.get('metrics',{})
    print(f"{d['factor_id']:26s} status={v.get('status','?'):12s} last={v.get('last_validated','?')} ic={m.get('ic'):.4f} icir={m.get('icir'):.3f} expr={d['calculation']['expression'][:70]}")