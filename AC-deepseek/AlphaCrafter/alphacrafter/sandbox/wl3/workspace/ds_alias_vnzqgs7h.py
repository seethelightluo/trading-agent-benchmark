import json
with open('factors/cn10y_beta_60.json') as f:
    d = json.load(f)
print(json.dumps({k: (v if k!='validation' else '...') for k,v in d.items() if k!='validation'}, indent=1)[:2500])
print('=== validation ===')
print(json.dumps(d['validation'], indent=1)[:1500])
