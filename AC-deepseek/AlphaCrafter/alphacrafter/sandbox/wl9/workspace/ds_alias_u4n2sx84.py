import os, json
print("=== Effective factors currently in library ===")
for f in sorted(os.listdir('factors/')):
    if not f.endswith('.json') or f=='factor_ensemble.json': continue
    try:
        d=json.load(open('factors/'+f))
        st=d.get('validation',{}).get('status')
        ic=d.get('validation',{}).get('metrics',{}).get('ic')
        print(f, "| status:", st, "| ic:", ic)
    except Exception as e:
        print(f, "ERR", e)