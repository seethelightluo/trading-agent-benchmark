import json
with open('factors/factor_ensemble.json') as f:
    ens = json.load(f)
print(json.dumps(ens, indent=2)[:3000])