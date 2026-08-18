import json
a = json.load(open('factor_ensemble.json'))['selected_factors']
b = json.load(open('factors/factor_ensemble.json'))['selected_factors']
print('root == factors dir:', a == b)
print('n factors:', len(a))
print('sum weights:', round(sum(x['weight'] for x in a), 6))
print('ids:', [x['factor_id'] for x in a])