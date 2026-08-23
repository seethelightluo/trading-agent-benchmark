import json, os
print(json.dumps(json.load(open('factors/factor_ensemble.json')), indent=1))
print("---factor files---")
for f in sorted(os.listdir('factors')):
    if f.endswith('.json') and not f.endswith('.bak'):
        print(f)