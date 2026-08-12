import json, os
for d in ['evicted','quarantine','rejected']:
    base = 'factors/'+d
    if not os.path.isdir(base): continue
    for f in sorted(os.listdir(base)):
        if f.endswith('.reason.json'):
            try:
                r = json.load(open(os.path.join(base,f)))
                s = json.dumps(r)[:300]
                print(d, '/', f, '->', s)
            except Exception as e:
                print(d, '/', f, 'ERR', e)
