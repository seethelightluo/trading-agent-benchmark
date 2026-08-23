import json
with open('factors/factor_ensemble.json') as f:
    ens = json.load(f)
print(json.dumps(ens, indent=1)[:2000])
print('---factor files---')
import os
for fn in sorted(os.listdir('factors')):
    if fn.endswith('.json') and not fn.endswith('.bak'):
        print(fn)