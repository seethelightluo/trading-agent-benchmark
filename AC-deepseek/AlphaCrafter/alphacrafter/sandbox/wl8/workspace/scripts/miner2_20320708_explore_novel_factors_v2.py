"""
miner2_20320708_explore_novel_factors_v2.py
Explore novel factor candidates for 15-instrument cross-asset universe.
Current data through 2032-07-07.

Factor candidates:
A) Cross-asset breadth: fraction of assets with positive 10d return
B) Risk-adjusted reversal: -(1d/5d return) / 20d vol, then rank
C) Cross-sectional dispersion of momentum: std of 20d returns across assets
D) Macro beta to DXY/USDCNY (observation-only signals)
E) Vol of vol ratio: vol_10d / vol_60d
F) Price acceleration: (short_mom - long_mom) / vol
G) Skewness factor: skew of 30d returns
H) Max drawdown / vol: risk-reward measure
"""
import numpy as np
from scipy.stats import spearmanr

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO_SIGNALS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
HORIZON = 10
MIN_VALID = 8

print("="*70)
print("FACTOR MINER 2 -- Novel Factor Exploration (2032-07-08)")
print("="*70)

# ---- data loading ----
data = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(sym, 500)
    if df is not None and len(df) >= 200:
        data[sym] = df
    else:
        print(f"WARN: {sym} has {len(df) if df is not None else 0} days, skip")

macro_data = {}
for sym in MACRO_SIGNALS:
    df = get_index_daily_data(sym, 500)
    if df is not None and len(df) >= 200:
        macro_data[sym] = df
        print(f"MACRO {sym}: {len(df)} days loaded")
    else:
        print(f"MACRO WARN: {sym} insufficient: {len(df) if df is not None else 0}")

print(f"\nValid instruments: {len(data)}/15")

# Build common date index across assets
all_dates = None
for sym, df in data.items():
    dts = set(str(d)[:10] for d in df['date'].values)
    if all_dates is None:
        all_dates = dts
    else:
        all_dates &= dts

# Also intersect with macro
for sym, df in macro_data.items():
    dts = set(str(d)[:10] for d in df['date'].values)
    all_dates &= dts

common_dates = sorted(all_dates)
print(f"Common dates (incl macro): {len(common_dates)}, range {common_dates[0]} to {common_dates[-1]}")

date_to_idx = {d:i for i,d in enumerate(common_dates)}
n_dates = len(common_dates)
n_assets = len(data)
assets_list = sorted(data.keys())

# Build aligned matrices
close_mat = np.full((n_dates, n_assets), np.nan)
pct_mat = np.full((n_dates, n_assets), np.nan)

for j, sym in enumerate(assets_list):
    df = data[sym]
    for _, row in df.iterrows():
        d = str(row['date'])[:10]
        if d in date_to_idx:
            i = date_to_idx[d]
            close_mat[i,j] = row['close']
            p = row['pct_change']
            pct_mat[i,j] = p if not np.isnan(p) else 0.0

# Build macro matrices
macro_mat = {}
for m_sym, m_df in macro_data.items():
    m_mat = np.full(n_dates, np.nan)
    for _, row in m_df.iterrows():
        d = str(row['date'])[:10]
        if d in date_to_idx:
            i = date_to_idx[d]
            m_mat[i] = row['close']
    macro_mat[m_sym] = m_mat

print(f"Matrix shape: {close_mat.shape}, NaN in close: {np.isnan(close_mat).mean():.4f}")

# Forward returns (10d horizon)
fwd_ret = np.full((n_dates, n_assets), np.nan)
for j in range(n_assets):
    fwd_ret[:,j] = np.array([
        close_mat[i+10,j]/close_mat[i,j]-1 if i+10 < n_dates else np.nan
        for i in range(n_dates)
    ])

# Cross-sectional IC function
def cs_ic(factor, fwd, min_valid=MIN_VALID):
    ics = []
    dt_count = 0
    for t in range(n_dates):
        fv = factor[t]
        rv = fwd[t]
        valid = ~(np.isnan(fv) | np.isnan(rv))