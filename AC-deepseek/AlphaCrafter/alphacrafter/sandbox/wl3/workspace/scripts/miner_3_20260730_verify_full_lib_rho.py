"""Verify round-18 PASS candidates against the FULL persisted library artifacts
(16 effective .npy signal matrices), then persist if still admissible."""
import sys, json, glob, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid, signal_matrix,
                           factor_to_panel, VAL_START, VAL_END)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates", flush=True)

# load all effective library artifacts from factors/*.json + *_signal.npy
lib_ranks = {}
for f in glob.glob('factors/*.json'):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        art = d.get('signal_artifact')
        if not art:
            print(f"  {fid}: no signal_artifact -> skip", flush=True)
            continue
        arr = np.load('factors/' + art, allow_pickle=False)
        if arr.shape != (len(grid), len(WATCHLIST)):
            print(f"  {fid}: shape {arr.shape} != grid -> skip", flush=True)
            continue
        lib_ranks[fid] = arr
    except Exception as e:
        print(f"  {f}: ERR {e}", flush=True)
print(f"full library artifacts loaded: {len(lib_ranks)} -> {sorted(lib_ranks)}", flush=True)

def rank_matrix(arr):
    out = np.full(arr.shape, np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = np.isfinite(row)
        if valid.sum() >= 3:
            r = pd.Series(row[valid]).rank().values
            out[i, valid] = r
    return out

lib_ranks = {fid: rank_matrix(arr) for fid, arr in lib_ranks.items()}

def max_lib_corr(rank_m):
    best, best_id = 0.0, None
    for fid, lr in lib_ranks.items():
        corrs = []
        for t in range(len(grid)):
            x = rank_m[t]; y = lr[t]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xv = x[m]; yv = y[m]
                xc = xv - xv.mean(); yc = yv - yv.mean()
                den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                if den > 0:
                    corrs.append((xc * yc).sum() / den)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# recompute candidate panels
spx = prices['SPX']['close']

def make_er(w):
    def f(df, s):
        c = df['close']
        net = (c - c.shift(w)).abs()
        path = c.diff().abs().rolling(w).sum()
        return net / path.replace(0, np.nan)
    return f

def make_rev(w, skip):
    def f(df, s):
        c = df['close']
        return -(c.shift(skip) / c.shift(skip + w) - 1.0)
    return f

cands = {
    'er_20': make_er(20),
    'rev_250_20': make_rev(250, 20),
}
for fid, fn in cands.items():
    panel = factor_to_panel(fn, prices)
    rank_m = rank_matrix(signal_matrix(panel, grid))
    rho, lib_id = max_lib_corr(rank_m)
    print(f"{fid}: full-library max|rho| = {rho:.4f} ({lib_id})", flush=True)
    json.dump({'rho': rho, 'lib': lib_id},
              open(f'scripts/miner_3_20260730_rho_full_{fid}.json', 'w'), indent=1)
print("done", flush=True)
