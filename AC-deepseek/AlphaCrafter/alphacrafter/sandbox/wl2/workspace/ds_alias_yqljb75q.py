import json
d = json.load(open('factor_ensemble.json'))
print('keys:', list(d.keys()))
print(json.dumps(d, indent=1)[:3000])