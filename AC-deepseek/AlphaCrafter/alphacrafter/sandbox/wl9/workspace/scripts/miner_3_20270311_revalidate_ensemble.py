"""
Re-validate the 4 factors in the current ensemble:
1. mom_120d_skip5  (weight 0.375, dir 1)
2. vol_of_vol20x60 (weight 0.254, dir 1) - NOT in our library files, uses .bak
3. mom_10d_skip5   (weight 0.195, dir 1)
4. vix_beta_cond_60x20 (weight 0.176, dir -1)

Plus key auxiliary factors like kaufman_eff_20d, vol_z_20d, bb_width_20d, etc.

Validation period: 2026-08 through 2027-03 (recent 7 months)
Use 10-day forward return horizon (same as admission gates)
"""
import sys, json, os
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

# Get account to get watchlist
acc = get_account_dict()
watchlist = acc.get('watch_list', [])
print(f"Watchlist: {watchlist}")
print(f"Number of instruments: {len(watchlist)}")

# Fetch data for all instruments - get as much as possible for validation
data = {}
MIN_DAYS = 400
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=MIN_DAYS)
    if df is None or len(df) < 200:
        print(f"WARNING: {sym} has insufficient data ({len(df) if df is not None else 0} days)")
        continue
    df['symbol'] = sym
    df['date'] = pd.to_datetime(df['date'])
    data[sym] = df
    print(f"  {sym}: {len(df)} days, {df.date.iloc[0].date()} to {df.date.iloc[-1].date()}")

# Also fetch VIX, DXY, USDCNY from index_data for macro factors
index_dir = '../persistent/index_data/'
macro_data = {}
for fname in ['VIX.csv', 'DXY.csv', 'USDCNY.csv']:
    path = os.path.join(index_dir, fname)
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    # Filter to relevant date range
    df = df.sort_values('date')
    key = fname.replace('.csv','')
    macro_data[key] = df
    print(f"  {key}: {len(df)} rows, {df.date.iloc[0].date()} to {df.date.iloc[-1].date()}")

# Merge into panel: align by date
# Build a combined DataFrame with all symbols
all_dates = None
for sym, df in data.items():
    dates = set(df.date.values)
    if all_dates is None:
        all_dates = dates
    else:
        all_dates = all_dates & dates

all_dates = sorted(all_dates)
print(f"\nCommon trading days: {len(all_dates)}")
print(f"Date range: {all_dates[0].date()} to {all_dates[-1].date()}")

# Build panel: dates x assets x close
panel = {}
for sym, df in data.items():
    df = df.set_index('date')
    panel[sym] = df

# Compute forward returns (10-day)
prices = pd.DataFrame({sym: panel[sym]['close'] for sym in panel.keys()})
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()

# Macro series aligned
vix_series = macro_data['VIX'].set_index('date')['close'].sort_index()
vix_series.index = pd.to_datetime(vix_series.index)
dxy_series = macro_data['DXY'].set_index('date')['close'].sort_index()
dxy_series.index = pd.to_datetime(dxy_series.index)
usdcny_series = macro_data['USDCNY'].set_index('date')['close'].sort_index()
usdcny_series.index = pd.to_datetime(usdcny_series.index)

# Reindex all to common dates
common_dates = prices.index.intersection(vix_series.index).intersection(dxy_series.index).intersection(usdcny_series.index)
common_dates = common_dates.sort_values()
print(f"Common dates with macro: {len(common_dates)} ({common_dates[0].date()} to {common_dates[-1].date()})")

prices = prices.loc[common_dates]
vix_series = vix_series.loc[common_dates]
dxy_series = dxy_series.loc[common_dates]
usdcny_series = usdcny_series.loc[common_dates]

# Forward returns
fwd_ret_10d = prices.pct_change(10).shift(-10)

# ============= FACTOR DEFINITIONS =============

def calc_mom_120d_skip5(prices):
    """close.shift(5) / close.shift(125) - 1.0"""
    return prices.shift(5) / prices.shift(125) - 1.0

def calc_mom_10d_skip5(prices):
    """close.shift(5) / close.shift(15) - 1.0"""
    return prices.shift(5) / prices.shift(15) - 1.0

def calc_vol_of_vol20x60(prices):
    """std of returns