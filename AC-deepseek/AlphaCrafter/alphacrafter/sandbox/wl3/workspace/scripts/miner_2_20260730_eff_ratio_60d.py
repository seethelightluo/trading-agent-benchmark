"""miner_2 2026-07-30 exploration: Kaufman efficiency ratio 60d.

Idea: trend persistence/smoothness. ER = |close - close.shift(n)| / sum(|ret|, n).
High ER -> smooth persistent trends; cross-sectional persistence should extend.
Expected direction +1.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel,
                           validate_factor, max_library_correlation,
                           LIBRARY_FACTORS, VAL_START, VAL_END)

prices = load_prices(days=2000)
print(f"Loaded {len(prices)} assets, dates {min(p['close'].index.min() for p in prices.values())}..{max(p['close'].index.max() for p in prices.values())}")

def eff_ratio_60(df, s):
    close = df['close']
    n = 60
    num = (close - close.shift(n)).abs()
    den = close.pct_change().abs().rolling(n).sum()
    return num / den

panel = factor_to_panel(eff_ratio_60, prices)
print(f"Factor panel shape: {panel.shape}, date range {panel.index.min()}..{panel.index.max()}")

# library panels
vix = load_index('VIX')
lib_panels = {}
def f_mom10(df, s): return df['close'].shift(5)/df['close'].shift(15)-1
def f_mom120(df, s): return df['close'].shift(5)/df['close'].shift(125)-1
def f_vixbeta(df, s):
    if vix is None: return None
    r = df['close'].pct_change(); vr = vix['close'].pct_change()
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v'])/z['v'].rolling(60).var()
    return -b*(vix['close']/vix['close'].shift(20)-1).reindex(z.index)
def f_vov(df, s): return df['close'].pct_change().rolling(20).std().rolling(60).std()
lib_panels['mom_10d_skip5'] = factor_to_panel(f_mom10, prices)
lib_panels['mom_120d_skip5'] = factor_to_panel(f_mom120, prices)
lib_panels['vix_beta_cond_60x20'] = factor_to_panel(f_vixbeta, prices)
lib_panels['vol_of_vol20x60'] = factor_to_panel(f_vov, prices)
for k, v in lib_panels.items():
    print(f"lib panel {k}: {v.shape}")

m = validate_factor('eff_ratio_60d', panel, prices)
print(json.dumps(m, indent=2, default=str))
if m is not None:
    rho, fid = max_library_correlation(panel, lib_panels)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = fid
    print(f"max_abs_library_correlation: {rho:.4f} vs {fid}")
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f} >= 0.0070? {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f} >= 0.084? {abs(m['icir'])>=0.084}")
