"""
Factor: Volatility-Adjusted Momentum (vol_adj_mom_20)
Idea: Scale raw 20-day return by its recent volatility to get a risk-adjusted
momentum signal. In high-vol regimes (VIX > 30), raw momentum becomes noisy;
adjusting by asset-specific vol should improve signal quality.

v2: Use 20-day return divided by 20-day volatility (annualized).
Higher values = strong risk-adjusted momentum.
"""

import numpy as np
import pandas as pd
import json
import sys
from datetime import datetime

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

symbols = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
           'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

LOOKBACK = 60  # minimum days needed
RET_DAYS = 20
VOL_DAYS = 20

# Gather data
data = {}
for s in symbols:
    df = get_stock_daily_data(s, 252)
    if df is not None and len(df) >= LOOKBACK:
        data[s] = df
    else:
        print(f"WARN: {s} insufficient data ({None if df is None else len(df)} days)")

N = len(data)
print(f"Symbols with sufficient data: {N}")
print(f"Date range: {min(df['date'].iloc[0] for df in data.values()).strftime('%Y-%m-%d')} -> "
      f"{max(df['date'].iloc[-1] for df in data.values()).strftime('%Y-%m-%d')}")

# Build panel: daily factor values and forward returns
all_dates = sorted(set(d for df in data.values() for d in df['date'].dt.date))
print(f"Total unique dates: {len(all_dates)}")

# For each date, compute factor = ret_20d / vol_20d
factor_records = []
forward_records = []

for i, date in enumerate(all_dates):
    date_dt = pd.Timestamp(date)
    factor_vals = {}
    valid_count = 0
    
    for s in symbols:
        if s not in data:
            continue
        df = data[s]
        # Find index of this date
        match_idx = df[df['date'] == date_dt].index
        if len(match_idx) == 0:
            continue
        idx = match_idx[0]
        
        # Need RET_DAYS before and VOL_DAYS before for computation
        if idx < max(RET_DAYS, VOL_DAYS):
            continue
        
        close_now = df.loc[idx, 'close']
        close_ret_ago = df.loc[idx - RET_DAYS, 'close']
        ret_20d = (close_now / close_ret_ago) - 1.0
        
        # 20-day volatility
        closes = df.loc[idx - VOL_DAYS: idx, 'close'].values
        daily_rets = np.diff(closes) / closes[:-1]
        vol_20d = np.std(daily_rets) * np.sqrt(252)
        
        if vol_20d <= 0 or np.isnan(vol_20d) or np.isnan(ret_20d):
            continue
        
        factor_val = ret_20d / vol_20d
        factor_vals[s] = factor_val
        valid_count += 1
    
    if valid_count < 8:  # minimum cross-section per the rules
        continue
    
    # Get forward returns (next 1 day, next 5 days, next 10 days)
    for fwd_days, label in [(1, 1), (5, 5), (10, 10)]:
        fwd_vals = {}
        fwd_valid = 0
        for s in data:
            df = data[s]
            match_idx = df[df['date'] == date_dt].index
            if len(match_idx) == 0:
                continue
            idx = match_idx[0]
            
            if idx + fwd_days >= len(df):
                continue
            
            close_now = df.loc[idx, 'close']
            close_fwd = df.loc[idx + fwd_days, 'close']
            fwd_ret = (close_fwd / close_now) - 1.0
            
            if np.isnan(fwd_ret):
                continue
            fwd_vals[s] = fwd_ret
            fwd_valid += 1
        
        if fwd_valid < 8:
            continue
        
        factor_records.append((date, factor_vals, label))
        forward_records.append((date, fwd_vals, label))

print(f"Total valid date-observations: {len(factor_records)}")

# Compute IC for each forward horizon
results = {}
for fwd_days in [1, 5, 10]:
    # Get factor-forward pairs for this horizon
    pairs = [(f, fw) for f, fw in zip(factor_records, forward_records) if f[2] == fwd_days and fw