import json, os
# Check remaining part of kaufman file structure and one more recent factor
with open('factors/skew_20d.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=1)[:1200])
print('...')
print(list(d['validation']['metrics'].keys()))