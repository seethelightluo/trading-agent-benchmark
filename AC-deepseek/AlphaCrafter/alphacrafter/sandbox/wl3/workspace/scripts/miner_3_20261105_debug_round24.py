"""Debug round-24 candidates that returned insufficient data / crashed."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           factor_to_panel, validate_factor, forward_returns,
                           rank_ic_series, signal_matrix, VAL_START, VAL_END)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()}", flush=True)

spx_r = prices['SPX']['close'].pct_change()
ndx_r = prices['NDX']['close'].pct_change()
btc_r = prices['BTC']['close'].pct_change(); eth_r = prices['ETH']['close'].pct_change()

def rb(r, m, w):
    return r.rolling(w).cov(m) / m.rolling(w).var().replace(0, np.nan)

def cond_beta(r, m, cond, w):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    z = z[cond(z['m'])]
    b = z['r'].rolling(w).cov(z['m']) / z['m'].rolling(w).var().replace(0, np.nan)
    return b.reindex(r.index)

def f_eth_btc_ratio_beta_60(df, s):
    return rb(df['close'].pct_change(), eth_r - btc_r, 60)

def f_beta_asym_60(df, s):
    r = df['close'].pct_change()
    db = cond_beta(r, spx_r, lambda m: m < 0, 60)
    tb = rb(r, spx_r, 60)
    return db / tb.replace(0, np.nan)

def f_ndx_spx_ratio_beta_60(df, s):
    return rb(df['close'].pct_change(), ndx_r - spx_r, 60)

def f_cvar_term_20_60(df, s):
    r = df['close'].pct_change()
    def cvar(win):
        q = r.rolling(win).quantile(0.05)
        return r.where(r <= q).rolling(win).mean().abs()
    return cvar(20) / cvar(60).replace(0, np.nan)

for fid, fn in [('eth_btc_ratio_beta_60', f_eth_btc_ratio_beta_60),
                ('beta_asym_60', f_beta_asym_60),
                ('ndx_spx_ratio_beta_60', f_ndx_spx_ratio_beta_60),
                ('cvar_term_20_60', f_cvar_term_20_60)]:
    panel = factor_to_panel(fn, prices)
    print(f"\n{fid}: panel {panel.shape} rows {len(panel)} range {panel.index.min()}..{panel.index.max()}", flush=True)
    print("  valid cells per asset:", panel.notna().sum().to_dict(), flush=True)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print("  validate_factor -> None", flush=True)
    else:
        print(f"  IC10={m['ic']:.4f} ICIR={m['icir']:.4f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f}", flush=True)
