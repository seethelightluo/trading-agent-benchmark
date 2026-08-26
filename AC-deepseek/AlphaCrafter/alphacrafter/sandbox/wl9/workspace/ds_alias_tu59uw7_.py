import json
# Show structure of one existing factor file (small, e.g. vix_roc or streak)
import os
files = sorted(f for f in os.listdir('factors') if not f.endswith('.bak'))
print('factor count:', len(files))
for f in files[:3]:
    d = json.load(open('factors/'+f))
    print('====', f)
    print(list(d.keys()))
    print(json.dumps({k: v for k, v in d.items() if k != 'validation' and k != 'signal_artifact'}, indent=1)[:1200])