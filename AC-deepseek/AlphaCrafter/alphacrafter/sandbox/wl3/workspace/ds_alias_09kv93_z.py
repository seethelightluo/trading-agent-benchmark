import json
print(open('factor_ensemble.json').read())
print("---PERSISTENT---")
import os
for d in ['../persistent', '../persistent/index_data']:
    if os.path.isdir(d):
        print(d, sorted(os.listdir(d))[:50])
    else:
        print(d, "MISSING")