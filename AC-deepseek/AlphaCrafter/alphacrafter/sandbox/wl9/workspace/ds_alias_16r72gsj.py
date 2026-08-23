import json, glob
for f in sorted(glob.glob('factors/*.json')):
    if '.bak' in f: continue
    try:
        d=json.load(open(f))
        print(f"{f.split('/')[-1]:28s} dir={d.get('expected_direction')} v={json.dumps(d.get('validation',{})).replace('\"','')[:230]}")
    except Exception as e:
        print(f, 'ERR', e)