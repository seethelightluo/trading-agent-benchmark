import json
with open('../persistent/date.json') as f:
    print(json.dumps(json.load(f), indent=2))