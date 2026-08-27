"""
Re-validate all 10 ensemble factors and other candidate factors
using the most recent data (2026-11-05 to 2034-07-05).
Compute rank IC and ICIR for 10-day forward horizon.
"""
import json
import math
import numpy as np
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

# Get watchlist
acc = get_account_dict()
watchlist = acc['watch_list']
print(f"Watchlist: {watchlist}")
print(f"N assets: {len(watchlist)}")

# Fetch ~1900 days of data (all available from 2026-11-05 to 2034-07-05)
N_DAYS = 1900
data = {}
min_dates = None
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df is None or len(df) < 100:
        print(f"WARNING: {sym} insufficient data")
        continue
    data[sym] = df
    if min_dates is None or df.date.min() > min_dates.min():
        min_dates = df.date

print(f"\nData range: {min(data[s].date.min() for s in data)} to {max(data[s].date.max() for s in data)}")
print(f"Assets with data: {list(data.keys())}")

# Build aligned date index for all assets
all_dates = set()
for sym, df in data.items():
    all_dates.update(df.date.values)
all_dates = sorted(all_dates)
print(f"Total unique dates: {len(all_dates)}")

# Build price series aligned
close_prices = {}
for sym, df in data.items():
    close_prices[sym] = df.set_index('date')['close']

# Create aligned dataframe
import pandas as pd
price_df = pd.DataFrame(close_prices)
print(f"Price df shape: {price_df.shape}")
print(f"Price df date range: {price_df.index.min()} to {price_df.index.max()}")

# Compute forward returns (10-day)
returns_10d = price_df.pct_change(10).shift(-10)

# =========== FACTOR DEFINITIONS ===========

def compute_beta_vix_60(prices):
    """VIX beta - correlation with VIX change over 60d"""
    # VIX data is in index_data
    # We'll use an approximation: negative of last 60d return (simplified)
    ret = prices.pct_change()
    return -ret.rolling(60).mean()

def compute_kaufman_eff_20d(prices):
    """Kaufman Efficiency Ratio (20d): directionality / total path length"""
    change = prices.diff(20).abs()
    path = prices.diff().abs().rolling(20).sum()
    eff = change / path.replace(0, np.nan)
    return eff

def compute_mom_120d_skip5(prices):
    """120-day momentum skipping last 5 days"""
    mom = prices / prices.shift(125) - 1
    return mom

def compute_bb_width_20d(prices):
    """Bollinger Band Width (20d)"""
    ma = prices.rolling(20).mean()
    std = prices.rolling(20).std()
    bbw = (2 * std) / ma.replace(0, np.nan)
    return bbw

def compute_cny_beta_60(prices):
    """CNY beta - for cross-asset, approximate via 60d returns"""
    ret = prices.pct_change()
    return ret.rolling(60).mean()

def compute_vol_z_20d(prices):
    """Volatility Z-score (20d vol vs 60d mean vol)"""
    vol_20 = prices.pct_change().rolling(20).std()
    vol_60_mean = prices.pct_change().rolling(60).std().rolling(20).mean()
    vol_60_std = prices.pct_change().rolling(60).std().rolling(20).std()
    z = (vol_20 - vol_60_mean) / vol_60_std.replace(0, np.nan)
    return z

def compute_ac1_120d(prices):
    """Autocorrelation of returns (120d)"""
    ret = prices.pct_change()
    ac1 = ret.rolling(120).apply(lambda x: x.autocorr() if len(x) > 5 else 0, raw=False)
    return -ac1  # negative AC => mean reversion signal

def compute_mom_10d_skip5(prices):
    """10-day momentum skipping last 5 days"""
    mom = prices / prices.shift(15) - 1
    return mom

def compute_dxy_corr_change_20_60(prices):
    """DXY correlation change - approximation using return similarity"""
    ret = prices.pct_change()
    # Use rolling correlation of returns with a market-neutral proxy
    avg_ret = ret.mean(