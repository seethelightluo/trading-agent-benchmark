import json
d = json.load(open('factors/factor_ensemble.json'))
print(list(d.keys()))
print(json.dumps(d.get('selected_factors'), indent=1)[:1500])
print('---meta---')
print(json.dumps({k: v for k, v in d.items() if k != 'selected_factors'}, indent=1)[:1500])