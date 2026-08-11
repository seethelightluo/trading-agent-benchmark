import json, glob, os
print("--- ENSEMBLE ---")
print(open('factors/factor_ensemble.json').read()[:2000])
print("--- EFFECTIVE FACTORS ---")
for f in sorted(glob.glob('factors/*.json')):
    b = os.path.basename(f)
    if 'bak' in b or 'ensemble' in b or 'evicted' in b or 'quarantine' in b or 'deprecated' in b: continue
    try:
        d = json.load(open(f))
        st = d.get('validation',{}).get('status','?')
        m = d.get('validation',{}).get('metrics',{})
        print(f"{b:45s} {st:12s} ic={m.get('ic', m.get('mean_ic','?'))} icir={m.get('icir', m.get('ICIR','?'))} last={d.get('last_validated','?')}")
    except Exception as e:
        print(f, 'ERR', e)
print("--- EVICTED ---")
for f in sorted(glob.glob('factors/evicted/*')):
    print(os.path.basename(f))
print("--- QUARANTINE ---")
for f in sorted(glob.glob('factors/quarantine/*')):
    print(os.path.basename(f))
