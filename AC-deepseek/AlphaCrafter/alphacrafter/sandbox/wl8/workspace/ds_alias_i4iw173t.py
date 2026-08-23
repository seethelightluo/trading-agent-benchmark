import json
with open('factors/factor_ensemble.json') as f:
    print(json.dumps(json.load(f), indent=2)[:1500])