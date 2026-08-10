"""miner2: recompute pairwise signal correlations across the whole persisted factor
library and write valid JSON provenance metadata (max_abs_library_correlation,
sibling_signal_corr) into every persisted factor file.

Signal panels are reconstructed from REAL daily OHLC data; no fabricated metrics.
"""
import json, glob, os, sys, math
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from miner1_common import SYMBOLS, load_close

closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[idx >= pd.Timestamp('2021-01-01')]

CP = pd.DataFrame({s: closes[s]['close'].reindex(idx).astype(float) for s in SYMBOLS})
OP = pd.DataFrame({s: closes[s]['open'].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]['high'].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]['low'].reindex(idx).astype(float) for s in SYMBOLS})

# ---- reconstruct signal panels exactly as persisted ----
panels = {}
for nd in (1, 2, 3, 5):
    panels[f"rev_{nd}d"] = -np.log(CP / CP.shift(nd))
for nd in (1, 2, 3, 5):
    hmax = HP.rolling(nd).max()
    lmin = LP.rolling(nd).min()
    rng = (hmax - lmin).replace(0, np.nan)
    panels[f"nclv_{nd}d"] = -(CP - lmin) / rng
rng1 = (HP - LP).replace(0, np.nan)
panels["nbody_1d"] = -(CP - OP) / rng1
panels["id_rev_1d"] = -(CP / OP - 1.0)
panels["rev_1d_vs"] = -np.log(CP / CP.shift(1)) / (CP.pct_change().rolling(20).std() + 1e-12)
# miner3 aliases (identical signal definitions, opposite sign for clv)
panels["miner3_rev_1d"] = -CP.pct_change(1)
hmax5 = HP.rolling(5).max()
lmin5 = LP.rolling(5).min()
panels["miner3_clv_5d"] = (CP - lmin5) / (hmax5 - lmin5 + 1e-12)


def panel_corr(a, b):
    A = a.values.astype(float)
    B = b.values.astype(float)
    m = np.isfinite(A) & np.isfinite(B)
    n = int(m.sum())
    if n < 50:
        return np.nan
    x = A[m]
    y = B[m]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def panel_key(fname, fid):
    if fname.startswith('miner2_'):
        return fid.replace('miner2_20260715_', '')
    if fname.startswith('miner3_'):
        short = fid.replace('miner3_20260716_', '')
        if short == 'rev_1d':
            return 'miner3_rev_1d'
        if short == 'clv_5d':
            return 'miner3_clv_5d'
        return short
    return None


def clean(v):
    """convert non-finite floats to None so json.dump emits null (valid JSON)"""
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, 4)
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    return v


files = sorted(glob.glob('factors/*.json'))
mapping = []  # (file, factor_id, panel_key)
for f in files:
    d = json.load(open(f))
    fid = d['factor_id']
    pk = panel_key(f.split('/')[-1], fid)
    if pk is None or pk not in panels:
        print(f"!! no panel for {f} (id={fid}, key={pk})")
        continue
    mapping.append((f, fid, pk))

# pairwise correlation matrix (signed)
signed = {}
for i in range(len(mapping)):
    for j in range(i + 1, len(mapping)):
        fi, fidi, pki = mapping[i]
        fj, fidj, pkj = mapping[j]
        r = panel_corr(panels[pki], panels[pkj])
        pair = (fidi, fidj) if fidi < fidj else (fidj, fidi)
        signed[pair] = r

print(f"recomputed pairwise rho for {len(mapping)} persisted factors on {len(idx)} dates")

for f, fid, pk in mapping:
    d = json.load(open(f))
    rhos = []
    sib = {}
    for o, ofid, opk in mapping:
        if ofid == fid:
            continue
        pair = (fid, ofid) if fid < ofid else (ofid, fid)
        r = signed.get(pair, np.nan)
        if np.isfinite(r):
            rhos.append((abs(r), ofid, r))
            sib[f"{ofid}"] = round(float(r), 4)
        else:
            sib[f"{ofid}"] = None
    rhos.sort(reverse=True)
    maxrho = float(rhos[0][0]) if rhos else 0.0
    d.setdefault('validation', {}).setdefault('metrics', {})
    d['validation']['metrics']['max_abs_library_correlation'] = round(maxrho, 4)
    d['validation']['metrics']['sibling_signal_corr'] = clean(sib)
    d['validation']['metrics']['n_library_factors'] = len(mapping)
    d['validation']['metrics']['n_corr_dates'] = int(len(idx))
    with open(f, 'w') as fh:
        json.dump(d, fh, indent=1, allow_nan=False)
    # reload-verify
    chk = json.load(open(f))
    assert chk['factor_id'] == fid, f"id mismatch {f}"
    assert chk['validation']['metrics']['max_abs_library_correlation'] == round(maxrho, 4), f"rho mismatch {f}"
    print(f"  {fid:42s} max_abs_lib_rho={maxrho:+.4f} n_sib={len(sib)} OK")

print("ALL UPDATED + RELOAD VERIFIED")
