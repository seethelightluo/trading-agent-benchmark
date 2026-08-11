"""miner_1 2026-08-27 exploration batch C: untapped signal families.

Batch B showed beta-family saturation (mkt/wti/sys-corr all crowd vs sx5e_beta
and comm_basket_beta). Batch C targets families with at most one library member:
  - OBV slope 20d (volume-flow momentum; library has only amihud illiquidity)
  - Intraday kurtosis 20d (skew family productive; kurtosis-of-total failed)
  - Vol autocorrelation 5d/60d (vol dynamics; vol_of_vol productive)
  - US10Y beta 60d (rates vein has CN10Y; US10Y is distinct US rates)
  - Semi-vol ratio 20d (downside std / upside std; gain_loss_asym was mean-based)
  - Cross-asset correlation centrality 60d (avg corr vs non-equity block
    XAU,COPPER,WTI,BTC,ETH,US10Y,CN10Y; avoids equity-block sx5e crowding)
  - Down-tail frequency 60d (share of days ret < -1.5*std60)

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

NON_EQ = ['XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

# ---- candidate factor functions ----
def f_obv_slope_20(df, s):
    c = df['close']
    v = df['volume'].replace(0, np.nan) if 'volume' in df.columns else None
    if v is None or v.isna().all():
        return None
    r = c.pct_change()
    obv = (np.sign(r) * v).fillna(0.0).cumsum()
    vmean = v.rolling(20).mean()
    slope = obv.diff(20) / (vmean * 20.0 + 1e-9)
    return slope

def f_intraday_kurtosis_20(df, s):
    if 'open' not in df.columns:
        return None
    ir = df['close'] / df['open'] - 1.0
    return ir.rolling(20, min_periods=12).kurt()

def f_vol_autocorr_5_60(df, s):
    r = df['close'].pct_change()
    v = r.abs()
    out = pd.Series(np.nan, index=v.index)
    for lag in range(1, 6):
        ac = v.rolling(60, min_periods=30).corr(v.shift(lag))
        out = out.add(ac, fill_value=0)
    return out / 5.0

def f_us10y_beta_60(df, s):
    us10y_r = prices['US10Y']['close'].pct_change().rename('us10y')
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), us10y_r.rename('y')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()).reindex(z.index)

def f_semi_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0)
    dn = r.where(r < 0)
    up_std = up.rolling(20, min_periods=10).std()
    dn_std = dn.rolling(20, min_periods=10).std()
    return dn_std / (up_std + 1e-12)

def f_cross_asset_corr_60(df, s):
    r = df['close'].pct_change().rename(s)
    others = {}
    for o in NON_EQ:
        if o != s and o in prices:
            others[o] = prices[o]['close'].pct_change()
    if not others:
        return None
    z = pd.concat([r, pd.DataFrame(others)], axis=1).dropna()
    if z.shape[1] < 3:
        return None
    corr = z.rolling(60, min_periods=30).corr()
    cols = [c for c in z.columns if c != s]
    s_corr = pd.DataFrame(index=z.index, columns=cols, dtype=float)
    for c in cols:
        s_corr[c] = corr.xs(c, level=1)[s] if len(corr) else np.nan
    return s_corr.mean(axis=1).reindex(df.index)

def f_down_tail_freq_60(df, s):
    r = df['close'].pct_change()
    std = r.rolling(60, min_periods=30).std()
    thresh = -1.5 * std
    hit = (r < thresh).astype(float)
    return hit.rolling(60, min_periods=30).mean()

# ---- evaluate ----
candidates = [
    ('obv_slope_20', f_obv_slope_20),
    ('intraday_kurtosis_20', f_intraday_kurtosis_20),
    ('vol_autocorr_5_60', f_vol_autocorr_5_60),
    ('us10y_beta_60', f_us10y_beta_60),
    ('semi_vol_ratio_20', f_semi_vol_ratio_20),
    ('cross_asset_corr_60', f_cross_asset_corr_60),
    ('down_tail_freq_60', f_down_tail_freq_60),
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

with open('scripts/miner_1_20260827_results_batchC.json', 'w') as f:
    json.dump(results, f, default=str, indent=1)
print("\nDONE saved scripts/miner_1_20260827_results_batchC.json")
