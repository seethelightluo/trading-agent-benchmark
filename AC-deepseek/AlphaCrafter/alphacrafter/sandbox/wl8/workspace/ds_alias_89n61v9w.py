import json
d = json.load(open('factors/factor_ensemble.json'))
print(json.dumps(d, indent=1)[:2500])