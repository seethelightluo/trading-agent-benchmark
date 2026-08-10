import json, glob, collections

d = json.load(open('factors/rejected/miner2_20260715_nbody_1d.json'))
for k, v in d.items():
    if k == 'signal_artifact':
        s = str(v)
        print('signal_artifact | type:', type(v).__name__, '| len:', len(s))
        print('  head:', s[:300])
        print('  tail:', s[-200:])
    else:
        print(k, '|', str(v)[:240])

print('=' * 70)
rs = collections.Counter()
for f in glob.glob('factors/rejected/*.reason.json'):
    try:
        dd = json.load(open(f))
        rs[str(dd.get('reason', '?'))[:130]] += 1
    except Exception as e:
        rs['UNPARSEABLE ' + str(e)[:80]] += 1
print('REJECTED reasons:')
for k, v in rs.most_common(12):
    print(f'  {v:3d}  {k}')

rs = collections.Counter()
for f in glob.glob('factors/quarantine/*.reason.json'):
    try:
        dd = json.load(open(f))
        rs[str(dd.get('reason', '?'))[:130]] += 1
    except Exception as e:
        rs['UNPARSEABLE ' + str(e)[:80]] += 1
print('QUARANTINE reasons:')
for k, v in rs.most_common(12):
    print(f'  {v:3d}  {k}')
