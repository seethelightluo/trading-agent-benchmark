import json
with open('factors/spx_corr60.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=1)[:2500])