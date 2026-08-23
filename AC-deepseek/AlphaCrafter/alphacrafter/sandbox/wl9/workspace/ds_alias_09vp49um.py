python -c "
import json, os
# Check the evicted vol_of_vol
for f in os.listdir('factors/evicted'):
    if 'vol_of_vol' in f.lower() and f.endswith('.json'):
        with open(f'factors/evicted/{f}') as fh:
            data = json.load(fh)
        print(f'=== {f} ===')
        print(json.dumps(data, indent=2)[:500])
        print()
"