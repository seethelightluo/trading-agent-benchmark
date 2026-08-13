import json
d = json.load(open('factor_ensemble.json'))
print(json.dumps(d['selected_factors'], indent=1)[:2500])
print('---TRADER NOTE---')
print(d.get('trader_note', 'none'))