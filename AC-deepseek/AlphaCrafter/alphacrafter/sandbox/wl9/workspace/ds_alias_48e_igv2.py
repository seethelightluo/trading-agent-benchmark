import json
d = json.load(open('factors/beta_VIX_60.json'))
print(list(d.keys()))
print(json.dumps(d.get('validation', {}), indent=1)[:2000])
print('---calc---')
print(json.dumps(d.get('calculation', {}), indent=1)[:800])