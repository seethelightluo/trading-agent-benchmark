import json
d = json.load(open('factor_ensemble.json'))
print(json.dumps(d, indent=1)[2500:6500])
