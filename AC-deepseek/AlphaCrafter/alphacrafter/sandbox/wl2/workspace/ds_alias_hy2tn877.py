import json, glob, os
files = sorted(glob.glob('factors/*.json'))
eff, dep = [], []
for f in files:
    try:
        d = json.load(open(f))
        st = d.get('validation',{}).get('status')
        if st == 'EFFECTIVE': eff.append((d.get('factor_id'), d.get('last_validated'), d.get('validation',{}).get('metrics',{}).get('ic'), d.get('validation',{}).get('metrics',{}).get('icir')))
        elif st and 'DEPREC' in st: dep.append(d.get('factor_id'))
    except Exception as e:
        pass
print("EFFECTIVE factors (%d):" % len(eff))
for e in sorted(eff, key=lambda x: str(x[1])): print(" ", e)
print("DEPRECATED:", dep[:20])
