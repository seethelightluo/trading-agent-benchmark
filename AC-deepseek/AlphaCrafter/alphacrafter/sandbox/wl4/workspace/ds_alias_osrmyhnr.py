import json
print(json.dumps(json.load(open('factor_ensemble.json')), indent=1))
print("---FACTOR JSON FILES---")
import os
for f in sorted(os.listdir('factors/')):
    if f.endswith('.json') and not f.endswith('.bak') and not f.endswith('.reason.json'):
        try:
            d = json.load(open('factors/'+f))
            print(f, '| status:', d.get('validation',{}).get('status'), '| id:', d.get('factor_id'))
        except Exception as e:
            print(f, 'ERR', e)
