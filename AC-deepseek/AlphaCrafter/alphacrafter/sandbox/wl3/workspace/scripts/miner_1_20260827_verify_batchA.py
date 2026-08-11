"""miner_1 2026-08-27: full-library correlation + robustness verification for batch A candidates.

The quick screen only checked rho vs 4 core panels. The deterministic gate recomputes
pairwise rho against ALL persisted signal artifacts. This script rebuilds the candidate
panels and correlates each against every factor signal matrix in factors/ (excluding
evicted/) to pre-audit crowding.
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, load_index, factor_to_panel, validate_factor,
                           build_library_panels, max_library_correlation, WATCHLIST,
                           canonical_grid, signal_matrix)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"canonical grid: {len(grid)} dates {grid.min().date()}..{grid.max().date()}")

# observation signals
usdjpy = load_index('USDJPY', prices=prices)
us10y = prices['US10Y']; cn10y = prices['CN10Y']
spread = (us10y['close'] - cn10y['close']).rename('spread') if us10y is not None and cn10y is not None else None
spread_r = spread.pct_change().rename('spread_r') if spread is not None else None
hs300_r = prices['000300.SH']['close'].pct_change().rename('hs300')

def ew_ret(symbols, prices):
    df = None
    for s in symbols:
        r = prices[s]['close'].pct_change().rename(s)
        df = r if df is None else pd.concat([df, r], axis=1)
    return df.mean(axis=1).rename('ew')

comm_r = ew_ret(['XAU', 'COPPER', 'WTI'], prices)

def f_comm_basket_beta_60(df, s):
    if comm_r is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), comm_r.rename('y')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()).reindex(z.index)

def f_down_beta_hs300_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), hs300_r.rename('y')], axis=1).dropna()
    z = z[z['y'] < 0]
    if len(z) < 30:
        return pd.Series(np.nan, index=df.index)
    return (z['r'].rolling(60, min_periods=20).cov(z['y']) / z['y'].rolling(60, min_periods=20).var()).reindex(df.index)

def f_mom_x_er_20(df, s):
    c = df['close']
    mom = c / c.shift(20) - 1.0
    path = c.diff().abs().rolling(20).sum()
    net = (c - c.shift(20)).abs()
    er = net / path
    return mom * er

cands = {
    'comm_basket_beta_60': f_comm_basket_beta_60,
    'down_beta_hs300_60': f_down_beta_hs300_60,
    'mom_x_er_20': f_mom_x_er_20,
}

# load all persisted library signal artifacts (excluding evicted/)
lib_artifacts = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    fid = p.name.replace('_signal.npy', '')
    lib_artifacts[fid] = np.load(p, allow_pickle=False)
print(f"loaded {len(lib_artifacts)} library artifacts")

def full_lib_rho(cand_arr, lib_artifacts):
    """Mean daily cross-sectional Spearman rho between candidate matrix and each lib matrix."""
    out = {}
    for fid, arr in lib_artifacts.items():
        if arr.shape != cand_arr.shape:
            continue
        corrs = []
        for i in range(cand_arr.shape[0]):
            x = cand_arr[i]; y = arr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xs = pd.Series(x[m]).rank().values
                ys = pd.Series(y[m]).rank().values
                c = np.corrcoef(xs, ys)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            out[fid] = float(np.mean(corrs))
    return out

results = {}
for fid, fn in cands.items():
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient"); continue
    cand_arr = signal_matrix(panel, grid)
    rhos = full_lib_rho(cand_arr, lib_artifacts)
    top = sorted(rhos.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    rho_max = max(abs(v) for v in rhos.values()) if rhos else 0.0
    rho_max_id = max(rhos, key=lambda k: abs(rhos[k])) if rhos else None
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho_max < 0.5
    print(f"=== {fid}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.3f}")
    print(f"    max full-lib rho={rho_max:.3f} ({rho_max_id}) -> {'PASS' if ok else 'FAIL'}")
    for k, v in top:
        print(f"      rho({k})={v:.3f}")
    m['max_abs_library_correlation'] = rho_max
    m['max_corr_library_id'] = rho_max_id
    m['full_lib_rho_top5'] = {k: round(v, 4) for k, v in top}
    m['n_lib_compared'] = len(rhos)
    results[fid] = {'ok': ok, 'metrics': m}

with open('scripts/miner_1_20260827_results_batchA_full.json', 'w') as f:
    json.dump(results, f, default=str, indent=1)
print("\nDONE saved scripts/miner_1_20260827_results_batchA_full.json")
