"""miner_1 2026-08-27: fix autocorr_20 and run full-library artifact audit for
batch B passers (adx_14, mom_skew_change, hilo_vol_ratio_20, autocorr_20).
"""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"grid: {len(grid)} dates {grid.min().date()}..{grid.max().date()}")

def f_autocorr_20(df, s):
    r = df['close'].pct_change().values
    out = np.full(len(r), np.nan)
    for i in range(20, len(r)):
        w = r[i-20:i]
        if np.all(np.isfinite(w)) and w.std() > 0:
            out[i] = np.corrcoef(w[:-1], w[1:])[0, 1]
    return pd.Series(out, index=df.index)

def f_adx_14(df, s):
    up = df['high'].diff()
    dn = -df['low'].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(14).mean()

def f_mom_skew_change(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).skew() - r.rolling(20).skew()

def f_hilo_vol_ratio_20(df, s):
    c = df['close']
    rng = (c.rolling(20).max() - c.rolling(20).min()) / c
    vol = c.pct_change().rolling(20).std()
    return rng / vol

cands = {
    'autocorr_20': f_autocorr_20,
    'adx_14': f_adx_14,
    'mom_skew_change': f_mom_skew_change,
    'hilo_vol_ratio_20': f_hilo_vol_ratio_20,
}

# full persisted artifact library (root factors/, conservative: includes stale npy)
lib_artifacts = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    fid = p.name.replace('_signal.npy', '')
    lib_artifacts[fid] = np.load(p, allow_pickle=False)
print(f"loaded {len(lib_artifacts)} artifacts")

def full_lib_rho(cand_arr):
    out = {}
    for fid, arr in lib_artifacts.items():
        if arr.shape != cand_arr.shape:
            continue
        corrs = []
        for i in range(cand_arr.shape[0]):
            x = cand_arr[i]; y = arr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                c = np.corrcoef(pd.Series(x[m]).rank().values, pd.Series(y[m]).rank().values)[0, 1]
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
        print(f"{fid}: insufficient")
        continue
    cand_arr = signal_matrix(panel, grid)
    rhos = full_lib_rho(cand_arr)
    rho_max = max(abs(v) for v in rhos.values()) if rhos else 0.0
    rho_max_id = max(rhos, key=lambda k: abs(rhos[k])) if rhos else None
    gate = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    ok = gate and rho_max < 0.5
    top = sorted(rhos.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    print(f"=== {fid}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.3f}")
    print(f"    max full-lib rho={rho_max:.3f} ({rho_max_id}) -> {'PASS' if ok else 'FAIL'}")
    for k, v in top:
        print(f"      rho({k})={v:.3f}")
    m['max_abs_library_correlation'] = rho_max
    m['max_corr_library_id'] = rho_max_id
    m['full_lib_rho_top4'] = {k: round(v, 4) for k, v in top}
    m['n_lib_compared'] = len(rhos)
    results[fid] = {'ok': ok, 'metrics': m}

with open('scripts/miner_1_20260827_results_batchB_full.json', 'w') as f:
    json.dump(results, f, default=str, indent=1)
print("\nDONE saved scripts/miner_1_20260827_results_batchB_full.json")
