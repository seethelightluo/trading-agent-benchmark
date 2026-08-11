"""miner_1 2026-08-27 exploration batch D: last untested index exposures +
cross-sectional momentum constructions.

Library betas: spx, sx5e, hs300, cn10y, down, comm_basket, dxy/eurusd/vix-cond.
Untested: NDX (US tech), HSI (HK), XAU (gold). Also CS-demeaned momentum and
CS rank-change momentum (relative-strength families currently only pairwise:
spx_rel_mom_20, gold_rel_mom_20).

Admission: |IC|>=0.007, |ICIR|>=0.084, max full-lib rho < 0.5.
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           WATCHLIST, canonical_grid, signal_matrix)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"loaded {len(prices)} assets; canonical grid {len(grid)} dates")

ndx_r = prices['NDX']['close'].pct_change().rename('ndx')
hsi_r = prices['HSI']['close'].pct_change().rename('hsi')
xau_r = prices['XAU']['close'].pct_change().rename('xau')

# ---- candidate factor functions ----
def f_ndx_beta_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), ndx_r.rename('y')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()).reindex(z.index)

def f_ndx_beta_cond_60x20(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), ndx_r.rename('y')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()
    y_move = prices['NDX']['close'] / prices['NDX']['close'].shift(20) - 1.0
    return (b * y_move).reindex(z.index)

def f_hsi_beta_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), hsi_r.rename('y')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()).reindex(z.index)

def f_xau_beta_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), xau_r.rename('y')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()).reindex(z.index)

def f_cs_demean_mom_20(df, s):
    """Per-asset 20d return minus the cross-sectional median 20d return."""
    mom = df['close'] / df['close'].shift(20) - 1.0
    return mom  # median demean applied at panel level

def f_cs_rank_mom_20(df, s):
    """Change in cross-sectional percentile rank of 20d return over 20 days."""
    mom = df['close'] / df['close'].shift(20) - 1.0
    return mom  # rank-diff applied at panel level

def _cs_demean(panel):
    med = panel.median(axis=1)
    return panel.sub(med, axis=0)

def _rank_mom(panel):
    r20 = panel
    ranked = panel.rank(axis=1, pct=True)
    return ranked - ranked.shift(20)

# ---- evaluate ----
candidates = [
    ('ndx_beta_60', f_ndx_beta_60),
    ('ndx_beta_cond_60x20', f_ndx_beta_cond_60x20),
    ('hsi_beta_60', f_hsi_beta_60),
    ('xau_beta_60', f_xau_beta_60),
    ('cs_demean_mom_20', f_cs_demean_mom_20),
    ('cs_rank_mom_20', f_cs_rank_mom_20),
]

lib_artifacts = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    lib_artifacts[p.name.replace('_signal.npy', '')] = np.load(p, allow_pickle=False)
print(f"loaded {len(lib_artifacts)} library artifacts for rho audit")

def full_lib_rho(cand_arr, lib_artifacts):
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
for fid, fn in candidates:
    try:
        panel = factor_to_panel(fn, prices)
        if fid == 'cs_demean_mom_20':
            panel = _cs_demean(panel)
        elif fid == 'cs_rank_mom_20':
            panel = _rank_mom(panel)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: insufficient -> None")
            results[fid] = {'ok': False, 'metrics': {'error': 'insufficient'}}
            continue
        cand_arr = signal_matrix(panel, grid)
        rhos = full_lib_rho(cand_arr, lib_artifacts)
        rho_max = max(abs(v) for v in rhos.values()) if rhos else 0.0
        rho_max_id = max(rhos, key=lambda k: abs(rhos[k])) if rhos else None
        top = sorted(rhos.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho_max < 0.5
        print(f"=== {fid}: panel {panel.shape} | IC={m['ic']:.4f} ICIR={m['icir']:.4f} "
              f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
              f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.3f}")
        print(f"    max full-lib rho={rho_max:.3f} ({rho_max_id}) -> {'PASS' if ok else 'FAIL'}")
        for k, v in top:
            print(f"      rho({k})={v:.3f}")
        print("    decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})
        m['max_abs_library_correlation'] = rho_max
        m['max_corr_library_id'] = rho_max_id
        m['full_lib_rho_top4'] = {k: round(v, 4) for k, v in top}
        results[fid] = {'ok': ok, 'metrics': m}
    except Exception as e:
        print(f"{fid}: ERROR {e}")
        results[fid] = {'ok': False, 'metrics': {'error': str(e)}}

with open('scripts/miner_1_20260827_results_batchD.json', 'w') as f:
    json.dump(results, f, default=str, indent=1)
print("\nDONE saved scripts/miner_1_20260827_results_batchD.json")
