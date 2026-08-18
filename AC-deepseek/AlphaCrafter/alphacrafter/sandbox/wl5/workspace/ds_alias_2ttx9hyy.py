import json, os
# Check evicted reasons
for f in sorted(os.listdir('factors/evicted')):
    if f.endswith('.reason.json'):
        with open(f'factors/evicted/{f}') as fp:
            r = json.load(fp)
        print(f, '->', json.dumps(r)[:220])
print()
# Check if any current factor has signal_artifacts key
for f in sorted(os.listdir('factors')):
    if f.endswith('.json') and not f.endswith('.bak') and f != 'factor_ensemble.json':
        with open(f'factors/{f}') as fp:
            d = json.load(fp)
        if 'signal_artifacts' in d:
            print(f, 'HAS signal_artifacts:', type(d['signal_artifacts']), str(d['signal_artifacts'])[:100])
        else:
            print(f, 'NO signal_artifacts; keys:', list(d.keys())[:12])