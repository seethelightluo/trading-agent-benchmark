import json
print(json.load(open('factors/factor_ensemble.json')))
import os
for d in ['evicted','quarantine','rejected']:
    print(d, os.listdir('factors/'+d) if os.path.isdir('factors/'+d) else 'MISSING')
