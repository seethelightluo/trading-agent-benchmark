import json
d = json.load(open('factors/factor_ensemble.json'))
print(json.dumps(d, indent=1)[:3000])
print('---FACTOR FILES---')
import glob, os
for f in sorted(glob.glob('factors/*.json')):
    if 'ensemble' in f or f.endswith('.bak'): continue
    try:
        dd = json.load(open(f))
        v = dd.get('validation', {})
        m = v.get('metrics', {})
        print(os.path.basename(f), '| status:', v.get('status'), '| lv:', dd.get('last_validated', v.get('validated_at','?')), '| IC:', m.get('ic'), '| ICIR:', m.get('icir'))
    except Exception as e:
        print(os.path.basename(f), 'ERR', str(e)[:80])