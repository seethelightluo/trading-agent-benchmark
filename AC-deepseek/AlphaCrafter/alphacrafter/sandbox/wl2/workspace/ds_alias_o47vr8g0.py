import json
d = json.load(open('factor_ensemble.json'))
print(json.dumps(d, indent=1)[:2500])
print("----")
f = json.load(open('factors/calmness_20.json'))
print(json.dumps(f, indent=1)[:1800])
