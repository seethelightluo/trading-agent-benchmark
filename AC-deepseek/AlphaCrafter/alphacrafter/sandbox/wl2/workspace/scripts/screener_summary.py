import json, glob, os

rows = []
for p in sorted(glob.glob('factors/*.json')):
    b = os.path.basename(p)
    if b == 'factor_ensemble.json' or '.bak' in b:
        continue
    try:
        d = json.load(open(p))
    except Exception as e:
        rows.append({'factor_id': b, 'err': str(e)})
        continue
    v = d.get('validation', {})
    m = v.get('metrics', {})
    ba = d.get('benchmark_admission', {})
    row = {
        'factor_id': d.get('factor_id', b),
        'tags': ','.join(d.get('tags', [])),
        'exp_dir': d.get('expected_direction'),
        'last_validated': d.get('last_validated') or ba.get('last_validated') or v.get('last_validated'),
        'status': v.get('status'),
        'metrics_keys': list(m.keys()) if isinstance(m, dict) else str(m)[:100],
    }
    rows.append(row)

for r in rows:
    print(json.dumps(r, default=str))
