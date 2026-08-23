python -c "
import json
with open('factors/factor_ensemble.json') as f:
    e=json.load(f)
# Print structure summary to understand current file completeness
print('schema_version:', e.get('schema_version'))
print('method:', e.get('method'))
print('selected:', e.get('selected_factors'))
print('notes keys:', list(e['notes'].keys()))
"