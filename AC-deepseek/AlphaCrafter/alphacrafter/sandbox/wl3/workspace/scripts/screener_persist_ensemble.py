"""Screener: build and persist factor_ensemble.json (quality_ic_tilt, top-10)."""
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
    recs.append({'id': d['factor_id'], 'ic': ic, 'icir': icir, 'q': q})

recs.sort(key=lambda r: r['q'], reverse=True)
top = recs[:10]

total_q = sum(r['q'] for r in top)
ensemble = {
    'schema_version': 1,
    'selected_factors': [
        {'factor_id': r['id'], 'weight': round(r['q'] / total_q, 10), 'direction': 1 if r['ic'] >= 0 else -1}
        for r in top
    ],
    'method': 'quality_ic_tilt'
}

wsum = sum(f['weight'] for f in ensemble['selected_factors'])
print('Selected factors (top-10 by q=|IC|*|ICIR|):')
for f in ensemble['selected_factors']:
    print(f"  {f['factor_id']:<28} w={f['weight']:.10f} dir={f['direction']:+d}")
print(f"sum(w) = {wsum:.10f}")
print(f"excluded #11: {recs[10]['id']} q={recs[10]['q']:.5f} (vs #10 q={recs[9]['q']:.5f})")

for path in ['factor_ensemble.json', 'factors/factor_ensemble.json']:
    with open(path, 'w') as fh:
        json.dump(ensemble, fh, indent=2)
    print('persisted ->', path)
