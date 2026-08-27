"""
Re-validation of all 10 ensemble factors + 7 additional effective factors.
Current date: 2034-07-06. Data spans from 2020 to current.
Check if each factor still has predictive power (IC >= 0.0070, ICIR >= 0.0840)
"""
import json
import sys
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

# Get watchlist
acct = get_account_dict()
watch_list = acct.get('watch_list', ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'])

# Fetch VIX data (observation-only macro indicator)
def get_vix_data():
    df = get_index_daily_data(symbol='VIX', days=1500)
    return df

def get_dxy_data():
    df = get_index_daily_data(symbol='DXY', days=1500)
    return df

# Fetch all data
print("="*80)
print("RE-VALIDATION OF EXISTING FACTORS - 2034-07-06")
print("="*80)

# Fetch data for all watchlist assets
all_data = {}
min_days = 800
for sym in watch_list:
    df = get_stock_daily_data(symbol=sym, days=min_days)
    if df is not None and len(df) >= 250:
        all_data[sym] = df
    else:
        print(f"WARNING: {sym} has insufficient data ({len(df) if df is not None else 0} rows)")

print(f"\nAssets with sufficient data: {len(all_data)}/{len(watch_list)}")
print(f"Assets: {list(all_data.keys())}")

# Get VIX and DXY
vix_df = get_vix_data()
dxy_df = get_dxy_data()
print(f"VIX data: {len(vix_df) if vix_df is not None else 0} rows")
print(f"DXY data: {len(dxy_df) if dxy_df is not None else 0} rows")

# Build aligned DataFrame of close prices
close_df = pd.DataFrame({sym: df.set_index('date')['close'] for sym, df in all_data.items()})
close_df = close_df.sort_index()
print(f"Close data shape: {close_df.shape}, dates: {close_df.index[0]} to {close_df.index[-1]}")

# Build VIX series
vix_series = vix_df.set_index('date')['close'] if vix_df is not None else None
dxy_series = dxy_df.set_index('date')['close'] if dxy_df is not None else None

# Forward returns (10-day horizon for admission)
def compute_forward_returns(close, horizon=10):
    return close.shift(-horizon) / close - 1.0

fwd_ret_10 = compute_forward_returns(close_df, 10)
fwd_ret_5 = compute_forward_returns(close_df, 5)
fwd_ret_21 = compute_forward_returns(close_df, 21)

# ============================================================
# Factor library definitions
# ============================================================

def calc_beta_VIX_60(close_df, vix_series):
    """60-day VIX beta: cov(asset_ret, VIX_ret)/var(VIX_ret)"""
    ret = close_df.pct_change()
    vix_ret = vix_series.pct_change()
    window = 60
    # Align
    combined = pd.concat([ret, vix_ret.rename('VIX')], axis=1).dropna()
    result = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for i in range(window, len(combined)):
        slice_ = combined.iloc[i-window:i]
        vix_var = slice_['VIX'].var()
        if vix_var > 1e-10:
            betas = slice_.drop('VIX', axis=1).apply(lambda c: c.cov(slice_['VIX']) / vix_var)
            result.iloc[i] = betas
    return result

def calc_kaufman_eff_20d(close_df):
    """Kaufman efficiency ratio: abs(change)/sum(abs(diff)) over 20d"""
    window = 20
    change = close_df.diff(window).abs()
    volatility = close_df.diff().abs().rolling(window).sum()
    eff = change / volatility.replace(0, np.nan)
    return eff

def calc_mom_120d_skip5(close_df):
    """120-day momentum with 5-day skip: close.shift(5)/close.shift(125)-1"""
    return close_df.shift(5) / close_df.shift(125) - 1.0

def calc_mom_10d_skip5(close_df):
    """10-day momentum with 5-day skip"""
    return close_df.shift(5) / close_df.shift(15) - 1.0

def calc_bb_width_20d(close_df):
    """Bollinger Band Width (20,2): (upper-lower)/middle"""
    ma = close_df.rolling(20).mean()
    std = close_df.rolling(20).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    bbw = (upper - lower) / ma
    return bbw

def calc_cny_beta_60(close_df, dxy_series):
    """60-day CNY-beta (using DXY as proxy for CNY direction)"""
    ret = close_df.pct_change()
    dxy_ret = dxy_series.pct_change()
    window = 60
    combined = pd.concat([ret, dxy_ret.rename('DXY')], axis=1).dropna()
    result = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for i in range(window, len(combined)):
        slice_ = combined.iloc[i-window:i]
        dxy_var = slice_['DXY'].var()
        if dxy_var > 1e-10:
            betas = slice_.drop('DXY', axis=1).apply(lambda c: c.cov(slice_['DXY']) / dxy_var)
            result.iloc[i] = betas
    return result

def calc_vol_z_20d(close_df):
    """Z-score of 20-day volatility relative to its 60-day history"""
    vol_20 = close_df.pct_change().rolling(20).std()
    vol_60 = close_df.pct_change().rolling(60).std()
    # Use rolling z-score of vol_20 relative to its own history
    vol_mean = vol_20.rolling(60).mean()
    vol_std = vol_20.rolling(60).std()
    vz = (vol_20 - vol_mean) / vol_std.replace(0, np.nan)
    return vz

def calc_ac1_120d(close_df):
    """First-order autocorrelation of returns over 120 days"""
    ret = close_df.pct_change()
    window = 120
    result = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for i in range(window, len(ret)):