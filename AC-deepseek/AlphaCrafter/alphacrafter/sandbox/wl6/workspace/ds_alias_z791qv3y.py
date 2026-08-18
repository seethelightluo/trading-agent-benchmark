import json
with open('factor_ensemble.json') as f:
    print(json.dumps(json.load(f), indent=2)[:2000])