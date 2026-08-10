"""Round 19: re-validate round-18 PASS candidates (er_20, rev_250_20) with the
full battery incl. full-library artifact rho, then persist if admissible."""
import sys, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid, signal_matrix,
                           factor_to_panel, validate_factor, persist_factor, VAL_START, VAL_END)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()}", flush=True)

# ---- full library artifacts (16 effective) ----
def rank_matrix(arr):
    out = np.full(arr.shape, np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = np.isfinite(row)
        if valid.sum() >= 3:
            r = pd.Series(row[valid]).rank().values
            out[i, valid] = r
    return out

lib_ranks = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        art = d.get('signal_artifact')
        if not art:
            print(f"  {fid}: no artifact -> skip", flush=True)
            continue
        arr = np.load('factors/' + art, allow_pickle=False)
        if arr.shape != (len(grid), len(WATCHLIST)):
            print(f"  {fid}: shape mismatch -> skip", flush=True)
            continue
        lib_ranks[fid] = rank_matrix(arr)
    except Exception as e:
        print(f"  {f}: ERR {e}", flush=True)
print(f"library artifacts loaded: {len(lib_ranks)}", flush=True)

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

# ---- candidates ----
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
    'er_20': (make_er(20), 'efficiency_ratio_20', 'efficiency ratio: |close - close(t-20)| / sum(|daily move|, 20)', 'trend-strength/volatility-efficiency'),
    'rev_250_20': (make_rev(250, 20), 'reversal_250_skip20', 'negative 250-day return skipping last 20 days (long-term reversal)', 'mean-reversion'),
}

results = {}
for fid, (fn, fname, expr, tag) in cands.items():
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: validation returned None", flush=True)
        continue
    rank_m = rank_matrix(signal_matrix(panel, grid))
    rho, lib_id = max_lib_corr(rank_m)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = lib_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"\n{fid}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"coverage={m['coverage_asset_days']:.3f} turnover={m['turnover_10d_rank']:.2f} "
          f"rho={rho:.3f}({lib_id}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    print("  decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}, flush=True)
    if ok and rho < 0.5:
        path, arr = persist_factor(
            factor_id=fid, factor_name=fname, expression=expr,
            description=expr,
            dependencies=['close'],
            parameters={'lookback': 20, 'skip': 20 if fid.startswith('rev') else 0},
            expected_direction=1, panel=panel, metrics=m, tags=[tag, 'cross_asset'],
            grid=grid, prices=None, version='1.0.0', status='EFFECTIVE',
            regime_notes='Validated on 15-instrument cross-asset universe 2020-01-01..2026-07-15 incl. 2022 hikes, 2023-24 equity rally, 2025 crypto vol.',
            extra={'validation_timestamp': '2026-07-30T00:00:00Z', 'miner': 'miner_3'})
        print(f"  PERSISTED {path}", flush=True)
        # read-back verification
        chk = json.load(open(path))
        assert chk['factor_id'] == fid, 'id mismatch'
        assert chk['validation']['status'] == 'EFFECTIVE'
        assert abs(chk['validation']['metrics']['ic']) >= 0.007
        assert abs(chk['validation']['metrics']['icir']) >= 0.084
        art = chk.get('signal_artifact')
        assert art and np.load('factors/' + art).shape == (len(grid), len(WATCHLIST)), 'artifact bad'
        print(f"  READBACK OK: {fid} status={chk['validation']['status']} ic={chk['validation']['metrics']['ic']:.4f} rho={chk['validation']['metrics']['max_abs_library_correlation']:.4f}", flush=True)
    else:
        print(f"  NOT persisted (ok={ok}, rho<0.5={rho < 0.5})", flush=True)
    results[fid] = {'ic': m['ic'], 'icir': m['icir'], 'rho': rho, 'ok': bool(ok)}
    json.dump(results, open(f'scripts/miner_3_20260730_round19_results.json', 'w'), indent=1)

print("\nSUMMARY:", json.dumps(results, indent=1), flush=True)
print("done", flush=True)
