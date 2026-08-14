import json
d = json.load(open('factors/calmness_20.json'))
print(json.dumps(d, indent=1)[:2500])