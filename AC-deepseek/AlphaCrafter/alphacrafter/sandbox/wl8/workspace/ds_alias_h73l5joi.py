import json
d = json.load(open('factors/factor_ensemble.json'))
print(json.dumps(d['notes'], indent=1)[2500:4500])