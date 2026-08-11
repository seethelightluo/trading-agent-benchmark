"""miner_1 2026-08-27 exploration batch B: novel signal families avoiding
saturated veins (20d momentum variants, index betas, vol ratios).

Motivation: library now contains many beta factors (spx/sx5e/hs300/cn10y/
dxy/eurusd/vix/comm-basket/down) and momentum variants (spx_rel_mom_20,
gold_rel_mom_20, mom_skew_change). Untested exposures in batch B:
  - USDCNY conditional beta (China FX regime; DXY/EURUSD/USDJPY tested, CNY not)
  - Cross-asset market beta (beta to EW return of all 15 tradable assets)
  - Excess kurtosis 20d (tail thickness; skew family tested, kurtosis not)
  - WTI oil beta 60d (commodity beta family, single-commodity version)
  - Return autocorrelation 5d (trend persistence microstructure)
  - Systemic correlation centrality 60d (avg pairwise corr vs universe)
  - VWAP deviation 20d (volume-aware price level)

Full-library rho audit against ALL persisted signal artifacts (npy) is done
inline; admission requires |IC|>=0.007, |ICIR|>=0.084, max full-lib rho < 0.5.
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, load_index, factor_to_panel, validate_factor,
                           build_library_panels, WATCHLIST, canonical_grid, signal_matrix)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"loaded {len(prices)} assets; canonical grid {len(grid)} dates {grid.min().date()}..{grid.max().date()}")

# ---- observation signals ----
usdcny = load_index('USDCNY', prices=prices)
usdcny_r = usdcny['close'].pct_change().rename('usdcny') if usdcny is not None else None
print(f"USDCNY len={0 if usdcny is None else len(usdcny)}")

def ew_ret(symbols, prices):
    df = None
    for s in symbols:
        r = prices[s]['close'].pct_change().rename(s)
        df = r if df is None else pd.concat([df, r], axis=1)
    return df.mean(axis=1).rename('ew')

wti_r = prices['WTI']['close'].pct_change().rename('wti')
mkt_r = ew_ret(WATCHLIST, prices)  # cross-asset market portfolio

# ---- candidate factor functions ----
def f_usdcny_beta_cond_60x20(df, s):
    if usdcny_r is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), usdcny_r.rename('y')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()
    y_move = usdcny['close'] / usdcny['close'].shift(20) - 1.0
    return (b * y_move).reindex(z.index)

def f_mkt_beta_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), mkt_r.rename('y')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()).reindex(z.index)

def f_kurtosis_20(df, s):
    r = df['close'].pct_change()
    k = r.rolling(20, min_periods=12).kurt()
    return k

def f_wti_beta_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), wti_r.rename('y')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()).reindex(z.index)

def f_ret_autocorr_5(df, s):
    r = df['close'].pct_change()
    # average of lag-1..5 autocorrelations over rolling 60d window
    out = pd.Series(np.nan, index=r.index)
    for lag in range(1, 6):
        ac = r.rolling(60, min_periods=30).corr(r.shift(lag))
        out = out.add(ac, fill_value=0)
    return out / 5.0

def f_sys_corr_60(df, s):
    """Mean pairwise rolling (60d) correlation of this asset's return with
    every other tradable asset's return on the same day."""
    r = df['close'].pct_change().rename(s)
    other = {}
    for o in WATCHLIST:
        if o == s or o not in prices:
            continue
        other[o] = prices[o]['close'].pct_change()
    o = pd.DataFrame(other)
    z = pd.concat([r, o], axis=1).dropna()
    corr = z.rolling(60, min_periods=30).corr()
    # corr is MultiIndex (date, col); extract corr of s vs each other col
    cols = [c for c in z.columns if c != s]
    s_corr = pd.DataFrame(index=z.index, columns=cols, dtype=float)
    for c in cols:
        s_corr[c] = corr.xs(c, level=1)[s] if len(corr) else np.nan
    return s_corr.mean(axis=1).reindex(df.index)

def f_vwap_dev_20(df, s):
    c = df['close']
    v = df['volume'].replace(0, np.nan) if 'volume' in df.columns else None
    if v is None or v.isna().all():
        return None
    tp = c * v
    vwap = tp.rolling(20).sum() / v.rolling(20).sum()
    return c / vwap - 1.0

# ---- evaluate ----
candidates = [
    ('usdcny_beta_cond_60x20', f_usdcny_beta_cond_60x20),
    ('mkt_beta_60', f_mkt_beta_60),
    ('kurtosis_20', f_kurtosis_20),
    ('wti_beta_60', f_wti_beta_60),
    ('ret_autocorr_5', f_ret_autocorr_5),
    ('sys_corr_60', f_sys_corr_60),
    ('vwap_dev_20', f_vwap_dev_20),
]

# load all persisted library artifacts
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

with open('scripts/miner_1_20260827_results_batchB.json', 'w') as f:
    json.dump(results, f, default=str, indent=1)
print("\nDONE saved scripts/miner_1_20260827_results_batchB.json")
