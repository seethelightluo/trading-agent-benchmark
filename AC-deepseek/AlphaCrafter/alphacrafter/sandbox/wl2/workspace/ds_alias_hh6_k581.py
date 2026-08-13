import json
ens = json.load(open('factors/factor_ensemble.json'))
print("updated_at:", ens.get('updated_at'))
print("cycle:", ens.get('cycle'))
print("method:", ens.get('method'))
print("note:", ens.get('note'))
print("n selected:", len(ens.get('selected_factors',[])))
for f in ens['selected_factors']:
    print(f['factor_id'], f['weight'], f['direction'], f['ic'], f['icir'])
