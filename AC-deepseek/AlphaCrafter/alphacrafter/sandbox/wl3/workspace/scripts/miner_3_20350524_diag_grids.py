"""Diagnose signal artifact grids across effective factors to ensure new artifact matches."""
import json, glob
import numpy as np
from pathlib import Path

grids = {}
for fp in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(fp))
    except Exception:
        continue
    if d.get('validation', {}).get('status') != 'EFFECTIVE':
        continue
    art = d.get('signal_artifact')
    if not art:
        continue
    p = Path('factors') / str(art)
    if not p.exists():
        print(f"{d['factor_id']}: MISSING artifact {art}")
        continue
    arr = np.load(p, allow_pickle=False)
    g = d.get('signal_artifact_grid', {})
    key = (g.get('start'), g.get('end'), g.get('n_dates'), arr.shape)
    grids.setdefault(key, []).append(d['factor_id'])

for k, v in sorted(grids.items(), key=lambda x: -len(x[1])):
    print(f"grid start={k[0]} end={k[1]} n_dates={k[2]} shape={k[3]} -> {len(v)} factors: {sorted(v)}")
print(f"\ntotal effective factors with artifacts: {sum(len(v) for v in grids.values())}")
print(f"distinct grids: {len(grids)}")
