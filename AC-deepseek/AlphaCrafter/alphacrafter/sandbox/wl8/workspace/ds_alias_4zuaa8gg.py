import json, glob, os
for f in glob.glob('factors/evicted/*.reason.json')[:6]:
    try:
        d=json.load(open(f))
        print('===', os.path.basename(f))
        print(json.dumps(d, indent=1)[:800])
    except Exception as e:
        print(f, 'ERR', e)