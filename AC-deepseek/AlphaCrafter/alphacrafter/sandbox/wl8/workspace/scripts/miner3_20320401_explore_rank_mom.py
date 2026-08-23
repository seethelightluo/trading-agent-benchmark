"""
miner3_20320401_explore_rank_mom.py
Explore: Cross-sectional percentile rank momentum factor.
Idea: Instead of raw momentum (which existing mom_10d_skip5 uses), 
use the cross-sectional rank of 20d return. This is robust to crypto outliers 
and captures relative strength within the 15-asset universe.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, pearsonr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
N_ASSETS = len(WATCH)

# Fetch a large history for warm-up + online period
data = {}
for s in WATCH:
    df = get_stock_daily_data(s, 2000)
    if df is not None and len(df) >= 120:
        data[s] = df
    else:
        print(f"WARNING: {s} insufficient data ({len(df) if df is not None else 0})")

print(f"Loaded {len(data)}/{N_ASSETS} assets with sufficient history")

# Align dates: build a single DataFrame of close prices
close_df = pd.DataFrame()
for s in data:
    df = data[s].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    close_df[s] = df['close']

close_df = close_df.sort_index().dropna(axis=1, how='all')
print(f"Close price DataFrame: {close_df.shape[0]} dates x {close_df.shape[1]} assets")
print(f"Date range: {close_df.index[0]} to {