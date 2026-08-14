import json
d = json.load(open('../persistent/date.json'))
print(json.dumps(d, indent=1)[:1500])