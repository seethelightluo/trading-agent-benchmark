"""
Re-validate ensemble factors (mom_120d_skip5, mom_10d_skip5, vol_of_vol20x60, vix_beta_cond_60x20)
and explore a new cross-sectional reversal factor.
Date: 2027-08-12
"""
import json, sys, os, math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

acct = get_account_dict()
print(f"Watchlist: {acct.get('watch_list', WATCHLIST)}")

# Fetch data
data = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=1000)
    if df is not None and len(df) > 120:
        data[sym] = df

print(f"Data for {len(data)}/{len(WATCHLIST)} assets")

# Build aligned close DataFrame
close_df = pd.DataFrame({sym: df.set_index('date')['close'] for sym, df in data.items()})
close_df = close_df.sort_index()
print(f"Close: {close_df.shape}, {close_df.index.min().date()} to {close_df.index.max().date()}")

# Fetch VIX
vix_df = get_index_daily_data(symbol='VIX', days=1000)
vix_series = vix_df.set_index('date')