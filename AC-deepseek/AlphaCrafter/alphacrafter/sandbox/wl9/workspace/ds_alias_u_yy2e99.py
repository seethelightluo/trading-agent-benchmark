import json
e=json.load(open('factor_ensemble.json'))
w=sum(f['weight'] for f in e['selected_factors'])
print('n_factors=',len(e['selected_factors']),'sum_weights=',round(w,4))
# verify directions align with library expected_direction
ok=True
for f in e['selected_factors']:
    d=json.load(open(f'factors/{f["factor_id"]}.json'))
    if d['expected_direction']!=f['direction']:
        ok=False; print('MISMATCH',f['factor_id'])
print('directions_align=',ok)