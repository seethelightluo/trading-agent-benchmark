#!/usr/bin/env python
"""
Exploration script: Rolling Sharpe Momentum factor (risk-adjusted momentum).
Factor: (close[t-5]/close[t-25] - 1) / rolling_std(daily_returns, 20)

Motivation: Pure momentum (mom_10d_skip5, mom_120d_skip5) ignores volatility 
regime. When vol is high, momentum signals may be less reliable. By dividing 
by vol, we get risk-adjusted momentum that should be more stable across vol regimes.

Benchmark gates: |IC| >= 0.0070, |ICIR| >= 0.0840 (10d horizon)
"""

import json
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

CURRENT_DATE = "2027-08-12"
LOOKBACK_DAYS = 500
HOLDING_PERIOD = 10
MIN_ASSETS_FOR_IC = 8

print("=" * 60)
print("Rolling Sharpe Momentum Factor (Risk-Adjusted Momentum)")
print("=" * 60)
print(f"Current date: {CURRENT_DATE}")
print(f"Lookback: {LOOKBACK_DAYS}d | Horizon: {HOLDING_PERIOD}d")

# Get watchlist
acct = get_account_dict()
watch_list = acct.get("watch_list", [
    "000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
    "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"
])
print(f"\nWatchlist ({len(watch_list)} assets): {watch_list}")

# Fetch data
data = {}
for sym in watch_list:
    try:
        df = get_stock_daily_data(symbol=sym, days=LOOKBACK_DAYS)
        if df is not None and len(df) >= 60:
            data[sym] = df.sort_values('date')
    except:
        pass

print(f"Data fetched for {len(data)}/{len(watch_list)} assets")

# Build aligned close DataFrame
close_df = pd.DataFrame({sym: df.set_index('date')['close'] for sym, df in data.items()})
close_df = close_df.sort_index()

# Daily returns
returns_df = close_df.pct_change()

# ---- 1. Compute Rolling Sharpe Momentum Factor ----
num_window = 20
vol_window = 20
skip = 5

# Momentum: return over past 20 days, shifted by 5 to skip microstructure noise
# close[t-5] / close[t-25] - 1
mom_20 = close_df.pct_change(periods=num_window).shift(skip)

# Volatility: rolling 20-day std of daily returns
vol_20 = returns_df.rolling(window=vol_window).std()

# Rolling Sharpe = momentum / volatility (avoid div by zero)
factor_df = mom_20 / vol_20.replace(0, np.nan)

print(f"\nFactor DataFrame shape: {factor_df.shape}")
print(f"Date range: {factor_df.index.min()} to {factor_df.index.max()}")
print(f"\nFactor stats:")
print(factor_df.describe().to_string())

# ---- 2. Compute IC at holding period horizon ----
fwd_returns = close_df.pct_change(periods=HOLDING_PERIOD).shift(-HOLDING_PERIOD)

common_idx = factor_df.index.intersection(fwd_returns.index)
factor_aligned = factor_df.loc[common_idx]
fwd_aligned = fwd_returns.loc[common_idx]

print(f"\nAligned dates for IC computation: {len(common_idx)}")
print(f"Date range: {common_idx.min()} to {common_idx.max()}")

# Cross-sectional Spearman rank IC per date
ic_values = []
valid_dates = 0
for dt in common_idx:
    f_vals = factor_aligned.loc[dt]
    r_vals = fwd_aligned.loc[dt]
    mask = f_vals.notna() & r_vals.notna()
    if mask.sum() >= MIN_ASSETS_FOR_IC:
        valid_dates += 1
        f = f_vals[mask].rank()
        r = r_vals[mask].rank()
        if len(f) >= 3:
            rho = np.corrcoef(f, r)[0, 1]
            if not np.isnan(rho):
                ic_values.append(rho)

ic_series = pd.Series(ic_values)
mean_ic = ic_series.mean()
std_ic = ic_series.std()
icir = mean_ic / std_ic if std_ic > 0 else 0
ic_hit = (ic_series > 0).mean()

print(f"\n===== Factor Validation Results =====")
print(f"Number of IC observations: {len(ic_values)}")
print(f"Valid IC dates (>= {MIN_ASSETS_FOR_IC} assets): {valid_dates}")
print(f"Mean IC (Spearman):         {mean_ic:.6f}")
print(f"Std IC:                     {std_ic:.6f}")
print(f"ICIR:                       {icir:.6f}")
print(f"IC Hit Ratio (>0):          {ic_hit:.4f}")
print(f"Min IC:                     {ic_series.min():.6f}")
print(f"Max IC:                     {ic_series.max():.6f}")
print(f"Median IC:                  {ic_series.median():.6f}")

# Sub-period analysis
mid_point = common_idx[len(common_idx)//2]
early_mask = ic_series.index < mid_point
late_mask = ic_series.index >= mid_point
if early_mask.sum() > 0 and late_mask.sum() > 0:
    early_ic = ic_series[early_mask]
    late_ic = ic_series[late_mask]
    print(f"\nSub-period analysis (split at {mid_point.date()}):")
    print(f"  Early period IC: {early_ic.mean():.6f} (ICIR: {early_ic.mean()/early_ic.std():.6f}, n={len(early_ic)})")
    print(f"  Late period IC:  {late_ic.mean():.6f} (ICIR: {late_ic.mean()/late_ic.std():.6f}, n={len(late_ic)})")

# ---- 3. Asset-level coverage ----
covg = factor_df.notna().mean(axis=0)
print(f"\nCoverage per asset:")
for sym in factor_df.columns:
    print(f"  {sym:15s}: {covg[sym]:.2%}")

# ---- 4. Decay analysis ----
print(f"\nDecay analysis (IC by horizon):")
for h in [1, 2, 3, 5, 10, 20]:
    fwd_h = close_df.pct_change(periods=h).shift(-h)
    common_h = factor_df.index.intersection(fwd_h.index)
    ic_h = []
    for dt in common_h:
        f_vals = factor_df.loc[dt]
        r_vals = fwd_h.loc[dt]
        mask = f_vals.notna() & r_vals.notna()
        if mask.sum() >= MIN_ASSETS_FOR_IC:
            f = f_vals[mask].rank()
            r = r_vals[mask].rank()
            if len(f) >= 3:
                rho = np.corrcoef(f, r)[0, 1]
                if not np.isnan(rho):
                    ic_h.append(rho)
    if len(ic_h) > 0:
        ic_arr = np.array(ic_h)
        print(f"  Horizon {h:2d}d: IC={ic_arr.mean():.6f}, ICIR={ic_arr.mean()/ic_arr.std():.6f}, n={len(ic_arr)}")
    else:
        print(f"  Horizon {h:2d}d: N/A")

# ---- 5. Threshold check ----
print(f"\n===== Gate Check =====")
ic_pass = abs(mean_ic) >= 0.0070
icir_pass = abs(icir) >= 0.0840
print(f"|IC|  >= 0.0070: {abs(mean_ic):.6f} >= 0.0070 => {ic_pass}")
print(f"|ICIR| >= 0.0840: {abs(icir):.6f} >=