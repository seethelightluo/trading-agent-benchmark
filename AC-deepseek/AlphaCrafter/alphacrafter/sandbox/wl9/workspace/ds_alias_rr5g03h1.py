import json
d = json.load(open('factors/vix_roc_20d.json'))
print(json.dumps(d, indent=1))