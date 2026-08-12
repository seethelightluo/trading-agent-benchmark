import json
print(json.dumps(json.load(open('factor_ensemble.json')), indent=2))