"""miner_2 shared helpers: library-correlation audit against ALL effective factor artifacts."""
import json, glob
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, canonical_grid, signal_matrix

def load_effective_artifacts():
    """Return dict {factor_id: (grid_start, grid_end, n_dates, matrix T x 15)} for all EFFECTIVE factors with artifacts."""
    out = {}
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
            continue
        arr = np.load(p, allow_pickle=False)
        fid = d.get('factor_id')
        g = d.get('signal_artifact_grid', {})
        out[fid] = {'arr': arr, 'start': g.get('start'), 'end': g.get('end'), 'n': g.get('n_dates')}
    return out

def max_library_correlation(panel, artifacts, grid=None):
    """Mean daily cross-sectional Spearman rho of candidate panel vs each effective library artifact."""
    m = panel.reindex(grid)
    cand = m[WATCHLIST].values.astype(float) if grid is not None else panel[WATCHLIST].values.astype(float)
    best = 0.0; best_id = None; details = {}
    for fid, a in artifacts.items():
        lib = a['arr']
        if lib.shape != cand.shape:
            continue
        corrs = []
        for t in range(len(cand)):
            x = cand[t]; y = lib[t]
            ok = np.isfinite(x) & np.isfinite(y) & ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() >= 8:
                rx = pd.Series(x[ok]).rank().values
                ry = pd.Series(y[ok]).rank().values
                c = np.corrcoef(rx, ry)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            details[fid] = r
            if abs(r) > best:
                best = abs(r); best_id = fid
    return best, best_id, details
