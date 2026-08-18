import json, os
print(os.getcwd())
with open('factor_ensemble.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=2)[:3000])