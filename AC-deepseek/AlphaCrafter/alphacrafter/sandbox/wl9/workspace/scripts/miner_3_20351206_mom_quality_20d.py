"""
Factor: mom_quality_20d
Description: Risk-adjusted momentum = 20d return / 20d volatility (std of daily returns).
Captures momentum scaled by risk - high momentum with low vol is "quality" momentum.
All 15 cross-asset instruments.
"""

import sys
sys.path.insert(0, '..')

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd
import numpy as np
from datetime import datetime

current_date = datetime(2035, 12, 6)

watch_list = [
    "000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH",
    "HSI", "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"
]

n_days = 800

universe = {}
for sym in watch_list:
    df = get_stock_daily_data(symbol=sym, days=n_days)
    if df is not None and len(df) >= 60:
        universe[sym] = df

print(f"Current date: {current_date.date()}")
print(f"Valid assets: {len(universe)}/{len(watch_list)}")

# Build factor panel
factor_panel = {}
for sym, df in universe.items():
    df = df.copy()
    df['ret_20'] = df['close'].pct_change(20)
    # Daily vol over same 20d window
    df['daily_ret'] = df['close'].pct_change(1)
    df['vol_20'] = df['daily_ret'].rolling(20).std()
    # Risk-adjusted momentum
    vol_zero = df['vol_20'].replace(0, np.nan)
    df['mom_quality'] = df['ret_20'] / vol_zero
    df['fwd_10'] = df['close'].pct_change(10).shift(-10)
    factor_panel[sym] = df['mom_quality'].values

fwd_panel = {}
for sym, df in universe.items():
    df = df.copy()
    fwd_panel[sym] = df['close'].pct_change(10).shift(-10).values

factor_df = pd.DataFrame(factor_panel)
fwd_df = pd.DataFrame(fwd_panel)

min_valid = 8
valid_mask = factor_df.notna().sum(axis=1) >= min_valid
factor_df = factor_df[valid_mask]
fwd_df = fwd_df[valid_mask]

print(f"Factor panel rows: {len(factor_df)}")

# IC
ic_vals = []
for i in range(len(factor_df)):
    fac = factor_df.iloc[i]
    fwd = fwd_df.iloc[i]
    valid = fac.notna() & fwd.notna()
    if valid.sum() >= min_valid:
        ic = fac[valid].corr(fwd[valid], method='spearman')
        if not np.isnan(ic):
            ic_vals.append(ic)

ic_series = pd.Series(ic_vals)
ic_mean = ic_series.mean()
ic_std = ic_series.std()
icir = ic_mean / ic_std * (len(ic_series) ** 0.5) if ic_std > 0 else 0

print(f"\n--- mom_quality_20d Validation ---")
print(f"IC observations: {len(ic_series)}")
print(f"Mean IC: {ic_mean:.6f}")
print(f"Std IC:  {ic_std:.6f}")
print(f"ICIR:    {icir:.6f}")
print(f"IC > 0 ratio: {(ic_series > 0).mean():.4f}")

coverage = factor_df.notna().mean()
print(f"Mean coverage: {coverage.mean():.4f}")

# Turnover
factor_rank = factor_df.rank(axis=1) / max(factor_df.shape[1], 1)
if len(factor_rank) > 12:
    turnover = factor_rank.diff(10).abs().mean(axis=1).mean()
    print(f"10d rank turnover: {turnover:.4f}")

# Decay
print(f"\nDecay analysis:")
for h in [1, 3, 5, 10, 20]:
    fwd_h = {}
    for sym, df in universe.items():
        vals = df['close'].pct_change(h).shift(-h).values
        fwd_h[sym] = vals
    fwd_h_df = pd.DataFrame(fwd_h)
    min_rows = min(len(factor_df), len(fwd_h_df))
    ic_h = []
    for i in range(min_rows):
        fac = factor_df.iloc[i]
        fwd_row = fwd_h_df.iloc[i]
        valid = fac.notna() & fwd_row.notna()
        if valid.sum() >= min_valid:
            ic_v = fac[valid].corr(fwd_row[valid], method='spearman')
            if not np.isnan(ic_v):
                ic_h.append(ic_v)
    if ic_h:
        ic_h_mean = np.mean(ic_h)
        ic_h_std = np.std(ic_h) if len(ic_h) > 1 else 0
        ic_h_ir = ic_h_mean / ic_h_std * (len(ic_h) ** 0.5) if ic_h_std > 0 else 0
        print(f"  Horizon {h:2d}d: IC={ic_h_mean:.6f}, ICIR={ic_h_ir:.6f}, n={len(ic_h)}")

# Sub-period
n_total = len(ic_series)
half = n_total // 2
ic_first = ic_series.iloc[:half]
ic_second = ic_series.iloc[half:]
print(f"\nSub-period:")
if len(ic_first) > 1:
    print(f"  First half: IC={ic_first.mean():.6f}, ICIR={ic_first.mean()/ic_first.std()*(len(ic_first)**0.5):.6f}")
if len(ic_second) > 1:
    print(f"  Second half: IC={ic_second.mean():.6f}, ICIR={ic_second.mean()/ic_second.std()*(len(ic_second)**0.5):.6f}")

ic_gate = abs(ic_mean) >= 0.0070
icir_gate = abs(icir) >= 0.0840
print(f"\nAdmission:")
print(f"  |IC| >= 0.0070? {'PASS' if ic_gate else 'FAIL'} (|{ic_mean:.6f}|)")
print(f"  |ICIR| >= 0.0840? {'PASS' if icir_gate else 'FAIL'} (|{icir:.6f}|)")

if ic_gate and icir_gate:
    print(f"\n*** PASSES ***")
else:
    print(f"\n*** FAILS ***")