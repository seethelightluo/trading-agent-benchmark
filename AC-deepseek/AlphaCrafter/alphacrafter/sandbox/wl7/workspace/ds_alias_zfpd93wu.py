import json
d = json.load(open('factor_ensemble.json'))
print(json.dumps(d, indent=2))