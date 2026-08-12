import json, glob, os
files = sorted(glob.glob('factors/*.json'))
for f in files:
    try:
        d = json.load(open(f))
        v = d.get('validation', {})
        m = v.get('metrics', {})
        print(f"{os.path.basename(f):55s} id={d.get('factor_id','?'):30s} status={v.get('status','?'):12s} last_val={str(d.get('last_validated'))[:19]:20s} IC={m.get('ic')} ICIR={m.get('icir')}")
    except Exception as e:
        print(f, 'ERR', e)
