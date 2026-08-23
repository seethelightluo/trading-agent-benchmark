import json, os
# Check one factor file structure to understand persisted format
with open('factors/kaufman_eff_20d.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=1)[:1500])