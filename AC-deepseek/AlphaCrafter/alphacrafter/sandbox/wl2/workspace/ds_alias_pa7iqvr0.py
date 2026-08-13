import json, glob, os
ens = json.load(open('factors/factor_ensemble.json'))
print("ensemble keys:", list(ens.keys())[:10])
if 'selected_factors' in ens:
    for f in ens['selected_factors']:
        print(f)
elif 'factors' in ens:
    for f in ens['factors']:
        print(f)
