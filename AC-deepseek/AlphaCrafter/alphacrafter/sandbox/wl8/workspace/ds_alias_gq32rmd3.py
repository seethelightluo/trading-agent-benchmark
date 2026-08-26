import json
with open('factor_ensemble.json') as f:
    ens = json.load(f)
print(json.dumps(ens, indent=2)[:3000])
print("---ROOT-LIB---")
import os
print(os.listdir('factors/'))
print("---REJECTED---")
print(os.listdir('factors/rejected/')[-20:])