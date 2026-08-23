"""
Factor Exploration: Volatility Regime Correlation Shift
======================================================
Idea: Measure each asset's cross-sectional rank correlation with VIX changes.
When VIX rises (risk-off), some assets (XAU, US10Y, CN10Y) tend to rise while
others (SPX, SOX, NDX) fall. This factor captures the 20-day rolling Spearman
correlation of each asset's daily return with VIX daily pct_change.

Null: If correlation with VIX is strongly negative -> asset is risk-on (sells off when VIX spikes)
If correlation with VIX is strongly positive -> asset is safe-haven (rises when VIX spikes)

We then test the predictive power of this factor for forward 10-day returns.

Markets: Recent VIX~28 suggests elevated vol regime; this factor may help
identify which assets benefit vs suffer in continued vol.
"""

import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import json, os, sys
from datetime import datetime

# Get watchlist
acc = get_account_dict()
wl = acc.get('watch_list', [])
print(f"Watchlist ({len(wl)} instruments): {wl}")

# Load VIX data - read from index_data
import csv
vix_prices = {}
vix_dates = []
try:
    with open('../persistent/index_data/VIX.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row['date']
            vix_dates.append(dt)
            vix_prices[dt] = float(row['close'])
    print(f"Loaded VIX data: {len(vix_prices)} rows, {vix_dates[0]} to {vix_dates[-1]}")
except:
    print("Could not load VIX.csv directly, using API")
    vix_df = get_index_daily_data(symbol='VIX', days=2000)
    if vix_df is not None:
        vix_df['date_str'] = vix_df['date'].astype(str)
        vix_prices = dict(zip(vix_df['date_str'], vix_df['close']))
        print(f"Loaded VIX via API: {len(vix_prices)} rows")

# Load asset data - get 1000 days for all assets  
N_DAYS = 1500
data = {}
min_len = 999999
for sym in wl:
    df = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df is not None and len(df) > 100:
        df['date_str'] = df['date'].astype(str)
        data[sym] = df
        min_len = min(min_len, len(df))
        print(f"{sym}: {len(df)} rows, {df['date_str'].iloc[0]} to {df['date_str'].iloc[-1]}")
    else:
        print(f"{sym}: insufficient data")

print(f"\nMin data rows across assets: {min_len}")

# Align all data by date
all_dates = set()
for sym, df in data.items():
    all_dates.update(df['date_str'].tolist())

# Filter to dates where VIX exists
all_dates = sorted([d for d in all_dates if d in vix_prices])
print(f"Total aligned dates: {len(all_dates)}")

# Compute factor: 20-day rolling Spearman correlation of asset return vs VIX pct_change
WINDOW = 20
HORIZON = 10  #