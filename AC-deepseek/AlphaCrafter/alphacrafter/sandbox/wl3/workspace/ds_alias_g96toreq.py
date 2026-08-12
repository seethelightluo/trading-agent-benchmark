import json
ens = json.load(open('factors/factor_ensemble.json'))
print(json.dumps(ens, indent=2))
s = sum(x['weight'] for x in ens['selected_factors'])
print("SUM:", round(s,6), "N:", len(ens['selected_factors']))