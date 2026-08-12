import json
with open('factors/factor_ensemble.json') as f:
    ens = json.load(f)
print('keys:', list(ens.keys()))
print('last_updated:', ens.get('last_updated'))
print('n selected:', len(ens.get('selected_factors', [])))
for x in ens.get('selected_factors', []):
    print(x.get('factor_id'), x.get('weight'), x.get('direction'), x.get('ic'), x.get('icir'), x.get('last_validated'))
