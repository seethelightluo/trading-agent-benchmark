import json, os
for f in sorted(os.listdir('factors/quarantine')):
    if f.endswith('.reason.json'):
        with open(f'factors/quarantine/{f}') as fp:
            r = json.load(fp)
        print(f, '->', json.dumps(r)[:300])
        print()