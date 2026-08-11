"""Screener: recompute quality tilt q = |IC|*|ICIR| for all active factors."""
import json, glob, os

files = sorted(glob.glob('factors/*.json'))
files = [f for f in files if not f.endswith('.bak') and 'ensemble' not in f]

recs = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    m = d.get('validation', {}).get('metrics', {})
    ic = m.get('ic', 0)
    icir = m.get('icir', 0)
    q = abs(ic) * abs(icir)
    recs.append({
        'id': d['factor_id'],
        'name': d.get('factor_name', ''),
        'ic': ic, 'icir': icir, 'q': q,
        'dir': 1 if ic >= 0 else -1,
        'turn': m.get('turnover_10d_rank', float('nan')),
        'cov': m.get('coverage_dates_ge8', float('nan')),
        'hit': m.get('ic_hit_ratio', float('nan')),
        'maxcorr': m.get('max_abs_library_correlation', float('nan')),
        'tags': d.get('tags', [])
    })

recs.sort(key=lambda r: r['q'], reverse=True)

hdr = f"{'#':>2} {'factor':<28} {'ic':>8} {'icir':>7} {'q':>9} {'dir':>4} {'turn10d':>8} {'cov':>6} {'hit':>6} {'maxcorr':>7}"
print(hdr)
print('-' * len(hdr))
for i, r in enumerate(recs, 1):
    print(f"{i:>2} {r['id']:<28} {r['ic']:>8.4f} {r['icir']:>7.4f} {r['q']:>9.5f} {r['dir']:>4} "
          f"{r['turn']:>8.2f} {r['cov']:>6.3f} {r['hit']:>6.3f} {r['maxcorr']:>7.3f}")

print()
print('=== Category mapping ===')
for r in recs:
    print(f"{r['id']:<28} tags={r['tags']}")

# Persist ranking for reproducibility
with open('scripts/_screener_rank.json', 'w') as fh:
    json.dump([{k: (v if not isinstance(v, float) or v == v else None) for k, v in r.items()} for r in recs], fh, indent=2)
print()
print('Ranking saved to scripts/_screener_rank.json')
