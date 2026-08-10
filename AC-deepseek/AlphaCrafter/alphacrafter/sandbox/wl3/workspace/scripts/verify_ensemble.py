import json, glob, os

active = [f for f in sorted(glob.glob('factors/*.json'))
          if not f.endswith('.bak') and os.path.basename(f) != 'factor_ensemble.json']

rows = []
for f in active:
    d = json.load(open(f))
    m = d['validation']['metrics']
    rows.append({
        'factor_id': d['factor_id'],
        'ic': m['ic'],
        'icir': m['icir'],
        'hit': m.get('ic_hit_ratio'),
        'q': abs(m['ic']) * abs(m['icir']),
        'dir': 1 if m['ic'] > 0 else -1,
        'max_rho': m.get('max_abs_library_correlation'),
        'max_rho_id': m.get('max_corr_library_id'),
        'turnover_10d_rank': m.get('turnover_10d_rank'),
    })

# Crowding exclusion: if max_rho > 0.7 with a higher-q partner, drop lower-q member
excluded = set()
for r in sorted(rows, key=lambda x: -x['q']):
    if r['max_rho'] and r['max_rho'] > 0.7:
        partner = r['max_rho_id']
        pq = next((x['q'] for x in rows if x['factor_id'] == partner), None)
        if pq is not None and pq > r['q']:
            excluded.add(r['factor_id'])

print(f"{'factor_id':<24}{'ic':>8}{'icir':>8}{'hit':>7}{'q':>9}{'dir':>5}{'maxrho':>8}  excl")
for r in sorted(rows, key=lambda x: -x['q']):
    print(f"{r['factor_id']:<24}{r['ic']:>8.4f}{r['icir']:>8.4f}{r['hit']:>7.3f}{r['q']:>9.4f}{r['dir']:>5}{r['max_rho'] if r['max_rho'] else 0:>8.3f}  {'*' if r['factor_id'] in excluded else ''}")

cand = [r for r in sorted(rows, key=lambda x: -x['q']) if r['factor_id'] not in excluded][:10]
tot = sum(r['q'] for r in cand)
print(f"\nSelected {len(cand)} factors, sum q = {tot:.6f}")
sel = []
for r in cand:
    w = r['q'] / tot
    sel.append({'factor_id': r['factor_id'], 'weight': round(w, 10), 'direction': r['dir'], 'q': r['q']})
    print(f"  {r['factor_id']:<24} w={w:.6f} dir={r['dir']:+d} q={r['q']:.4f}")
print(f"weights sum = {sum(s['weight'] for s in sel):.10f}")

# Compare to persisted
ens = json.load(open('factors/factor_ensemble.json'))
old = {s['factor_id']: (s['weight'], s['direction']) for s in ens['selected_factors']}
new = {s['factor_id']: (s['weight'], s['direction']) for s in sel}
print("\n== Diff vs persisted ensemble ==")
all_ids = set(old) | set(new)
for fid in sorted(all_ids):
    o, n = old.get(fid), new.get(fid)
    mark = 'SAME' if o == n else 'DIFF'
    print(f"  {fid:<24} old={o} new={n}  {mark}")

# Persist
out = {
    'schema_version': 1,
    'selected_factors': [{'factor_id': s['factor_id'], 'weight': s['weight'], 'direction': s['direction']} for s in sel],
    'method': 'quality_ic_tilt',
}
with open('factors/factor_ensemble.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print("\nPersisted factors/factor_ensemble.json")
