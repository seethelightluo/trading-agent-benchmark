import json
with open('factors/factor_ensemble.json') as f:
    e = json.load(f)
print('schema_version:', e.get('schema_version'))
print('method:', e.get('method'))
print('selected:', e.get('selected_factors'))
print('notes keys:', list(e.get('notes', {}).keys()))