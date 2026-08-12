import json, os
print("=== factor_ensemble.json (raw) ===")
print(open('factor_ensemble.json').read())
print()
# Look at a sample factor file
f = json.load(open('factors/cn10y_beta_60.json'))
print("=== sample factor cn10y_beta_60 ===")
print(json.dumps(f, indent=1)[:3000])