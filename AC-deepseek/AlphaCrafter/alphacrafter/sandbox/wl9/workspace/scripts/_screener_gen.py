import json, glob

rows = []
for p in glob.glob('factors/*.json'):
    if p.endswith('.bak') or '/evicted/' in p:
        continue
    try:
        d = json.load(open(p))
        b = d.get('benchmark_admission') or {}
        m = b.get('selected_metrics') or {}
        if m.get('ic') is None:
            continue
        rows.append((d['factor_id'], m['ic'], m['icir'], m['quality'],
                     d.get('expected_direction'),
                     m.get('reported_max_abs_library_correlation') or 0.0))
    except Exception:
        pass

rows.sort(key=lambda x: -x[3])  # quality desc
sel = rows[:10]  # max 10

tot = sum(r[3] for r in sel)
ens = []
for fid, ic, icir, q, direction, corr in sel:
    ens.append({
        'factor_id': fid,
        'weight': round(q / tot, 5),
        'direction': -1 if ic < 0 else 1,
    })

# normalize weights to sum exactly 1
s = sum(x['weight'] for x in ens)
for x in ens:
    x['weight'] = round(x['weight'] / s, 5)
# adjust last to reach 1.0 exactly
diff = 1.0 - sum(x['weight'] for x in ens)
ens[-1]['weight'] = round(ens[-1]['weight'] + diff, 5)

out = {'schema_version': 1, 'selected_factors': ens, 'method': 'quality_ic_tilt'}
json.dump(out, open('factor_ensemble.json', 'w'), indent=1)

print('selected', len(ens), 'wsum', round(sum(x['weight'] for x in ens), 5))
for x in ens:
    print(x['factor_id'], x['weight'], 'dir', x['direction'])