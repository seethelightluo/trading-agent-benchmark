"""miner_2 2030-09-19: explore a batch of novel factor candidates.

Each candidate is a distinct idea; we run the shared validation battery
(factor_common.validate_factor, horizon-10 IC on the warm-up window
2020-01-01..2026-07-15) and audit redundancy vs the whole persisted library
signal artifacts (canonical-grid .npy matrices -> mean daily Spearman rho).
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           validate_factor, canonical_grid, signal_matrix,
                           VAL_START, VAL_END)

prices = load_prices(days=3500)
max_date = max(dd.index.max() for dd in prices.values())
print(f"data max_date = {max_date.date()}, n_assets = {len(prices)}")

spx = prices['SPX']['close']
btc = prices['BTC']['close']
us10y = prices['US10Y']['close']
vix = load_index('VIX', prices=prices)
dxy = load_index('DXY', prices=prices)

# ----------------------------------------------------------------------------
# Candidate factor definitions (one idea each)
# ----------------------------------------------------------------------------

def f_ups_beta_spread_60(df, s):
    """Convexity: (upside beta - downside beta) vs SPX over 60d.
    High value = asset rises more than it falls relative to SPX (positive convexity)."""
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), spx.pct_change().rename('m')], axis=1).dropna()
    up = z['m'] > 0
    dn = z['m'] < 0
    b_up = z['r'].rolling(60, min_periods=40).apply(
        lambda x: np.cov(x, z['m'].loc[x.index])[0, 1] / np.var(z['m'].loc[x.index]) if len(x) >= 40 and np.var(z['m'].loc[x.index]) > 0 else np.nan, raw=False) if False else None
    # simpler vectorized version
    cov_up = z['r'].where(up).rolling(60, min_periods=30).cov(z['m'].where(up))
    var_up = z['m'].where(up).rolling(60, min_periods=30).var()
    cov_dn = z['r'].where(dn).rolling(60, min_periods=30).cov(z['m'].where(dn))
    var_dn = z['m'].where(dn).rolling(60, min_periods=30).var()
    b_up = cov_up / var_up.replace(0, np.nan)
    b_dn = cov_dn / var_dn.replace(0, np.nan)
    return (b_up - b_dn).reindex(df.index)


def f_crypto_beta_60(df, s):
    """60d rolling beta of asset daily returns to BTC returns (cross-asset linkage)."""
    r = df['close'].pct_change()
    m = btc.pct_change()
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var().replace(0, np.nan)
    return b.reindex(df.index)


def f_eff_ratio_60(df, s):
    """Trend efficiency over 60d: |net move| / sum(abs(daily returns))."""
    c = df['close']
    r = c.pct_change()
    net = (c / c.shift(60) - 1.0).abs()
    path = r.abs().rolling(60).sum()
    return (net / path.replace(0, np.nan)).reindex(df.index)


def f_volume_trend_20_60(df, s):
    """Liquidity/participation expansion: 20d avg volume / 60d avg volume - 1."""
    if 'volume' not in df or df['volume'].notna().sum() < 100:
        return None
    v = pd.to_numeric(df['volume'], errors='coerce')
    return (v.rolling(20).mean() / v.rolling(60).mean() - 1.0).reindex(df.index)


def f_skew_60(df, s):
    """Realized skewness of daily returns over 60d."""
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=40).skew().reindex(df.index)


def f_bond_link_60(df, s):
    """60d rolling correlation of asset returns with US10Y price changes
    (bond price = 1/(1+yield) proxy -> -diff of yield). Assets that hedge rates."""
    r = df['close'].pct_change()
    bond_px = -us10y.diff()
    z = pd.concat([r.rename('r'), bond_px.rename('b')], axis=1).dropna()
    c = z['r'].rolling(60, min_periods=40).corr(z['b'])
    return c.reindex(df.index)


def f_dxy_sens_60(df, s):
    """60d rolling beta of asset returns to DXY changes (dollar sensitivity)."""
    if dxy is None:
        return None
    r = df['close'].pct_change()
    m = dxy['close'].pct_change()
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var().replace(0, np.nan)
    return b.reindex(df.index)


CANDIDATES = [
    ('ups_beta_spread_60', f_ups_beta_spread_60, 'convexity'),
    ('crypto_beta_60', f_crypto_beta_60, 'cross_asset'),
    ('eff_ratio_60', f_eff_ratio_60, 'trend'),
    ('volume_trend_20_60', f_volume_trend_20_60, 'liquidity'),
    ('skew_60', f_skew_60, 'distribution'),
    ('bond_link_60', f_bond_link_60, 'cross_asset'),
    ('dxy_sens_60', f_dxy_sens_60, 'macro'),
]

# ----------------------------------------------------------------------------
# Library redundancy audit: load all persisted signal artifacts on canonical grid
# ----------------------------------------------------------------------------
grid = canonical_grid(prices)
lib_art = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    fid = p.name.replace('_signal.npy', '')
    try:
        arr = np.load(p, allow_pickle=False)
        lib_art[fid] = arr
    except Exception:
        pass
print(f"library artifacts loaded: {len(lib_art)}")


def max_lib_corr(panel):
    mtx = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, arr in lib_art.items():
        if arr.shape != mtx.shape:
            continue
        corrs = []
        for i in range(len(grid)):
            x, y = mtx[i], arr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                from scipy.stats import spearmanr
                c = spearmanr(x[m], y[m]).statistic
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


results = {}
for fid, fn, tag in CANDIDATES:
    print(f"\n===== {fid} ({tag}) =====")
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient data -> None")
        results[fid] = None
        continue
    rho, rho_id = max_lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"panel {panel.shape}  dates {panel.index.min().date()}..{panel.index.max().date()}")
    print(f"IC10={m['ic']:+.4f}  ICIR10={m['icir']:+.3f}  hit={m['ic_hit_ratio']:.3f}  n={m['n_ic_dates']}")
    print(f"coverage={m['coverage_asset_days']:.3f}  ge8={m['coverage_dates_ge8']:.3f}  turnover={m['turnover_10d_rank']:.3f}")
    print(f"decay: " + json.dumps({k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}))
    print(f"max_lib_corr={rho:.3f} (vs {rho_id})")
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f}>=0.084 {abs(m['icir'])>=0.084} -> {'PASS' if ok else 'FAIL'}")
    results[fid] = m

print("\n===== SUMMARY =====")
for fid, m in results.items():
    if m is None:
        print(f"{fid:24s} INSUFFICIENT")
    else:
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"{fid:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.3f} rho={m['max_abs_library_correlation']:.3f} {'PASS' if ok else 'FAIL'}")
