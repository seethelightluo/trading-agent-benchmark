"""
Explore: Momentum Divergence Factor (20d vs 60d)
Compares short-term momentum (20d) with medium-term momentum (60d).
When ST mom > MT mom, asset is accelerating upward (buy signal).
When ST mom < MT mom, asset is decelerating (sell signal).
This is a trend-change detection factor.

Universe: 15 cross-asset instruments, validated across all available dates.
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json

CURRENT_DATE = "2027-05-06"
MIN_DAYS = 150  # need enough for 120d lookback

acc = get_account_dict()
watchlist = acc.get('watch_list', [])
print(f"Watchlist ({len(watchlist)}): {watchlist}")

# Fetch data for all instruments
data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=400)
    if df is not None and len(df) >= MIN_DAYS:
        data[sym] = df
    else:
        print(f"  WARN: {sym} insufficient data: {len(df) if df is not None else 0}")

print(f"\nLoaded {len(data)}/{len(watchlist)} instruments with sufficient data")

# Fetch VIX data for regime context
vix_df = get_index_daily_data(symbol="VIX", days=400)
if vix_df is not None:
    print(f"VIX data: {len(vix_df)} rows, current VIX = {vix_df['close'].iloc[-1]:.2f}")
else:
    print("WARN: No VIX data available")

# Build aligned dataframe of 20d and 60d returns
# 20d momentum: close[t] / close[t-20] - 1 (shift by 5 to avoid lookahead)
# 60d momentum: close[t] / close[t-60] - 1

factor_values_dict = {}
forward_returns_dict = {}

for sym, df in data.items():
    df = df.copy()
    df['ret_20d'] = df['close'].shift(5) / df['close'].shift(25) - 1.0  # 20d mom
    df['ret_60d'] = df['close'].shift(5) / df['close'].shift(65) - 1.0  # 60d mom
    # Momentum divergence: ST - MT (positive means accelerating)
    df['mom_div'] = df['ret_20d'] - df['ret_60d']
    
    # Forward 10-day return (holding period)
    df['fwd_10d'] = df['close'].shift(-10) / df['close'] - 1.0
    
    # Store by date
    for i in range(len(df)):
        date = df['date'].iloc[i]
        if pd.isna(date):
            continue
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        
        val = df['mom_div'].iloc[i]
        fwd = df['fwd_10d'].iloc[i]
        
        if not np.isnan(val) and not np.isnan(fwd):
            if date_str not in factor_values_dict:
                factor_values_dict[date_str] = {}
                forward_returns_dict[date_str] = {}
            factor_values_dict[date_str][sym] = val
            forward_returns_dict[date_str][sym] = fwd