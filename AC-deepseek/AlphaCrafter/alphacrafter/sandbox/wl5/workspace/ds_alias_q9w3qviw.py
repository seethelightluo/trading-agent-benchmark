import json
with open('factors/factor_ensemble.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=2)[:2500])