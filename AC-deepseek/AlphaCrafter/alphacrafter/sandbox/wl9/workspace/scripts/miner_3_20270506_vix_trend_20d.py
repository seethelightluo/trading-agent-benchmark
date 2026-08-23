"""
Factor: vix_trend_20d
Description: 20-day rate of change of VIX index.
Rationale: Direction of VIX change predicts risk-off/risk-on rotation.
When VIX is rising (positive trend), safe havens (XAU, US10Y, CN10Y) should 
outperform risk assets (SPX, NDX, SOX, etc.). When falling, risk assets bounce.
In VIX=36 environment, the *trend* of VIX matters more than the level.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

# Get watchlist and data
data = get_account_dict()
watchlist = data['watch_list']
print(f"Watchlist ({len(watchlist)}): {watchlist}")

# Get VIX data
vix_df = get_index_daily_data("VIX", 500)
print(f"VIX data: {len(vix_df)} days, {vix_df.date.iloc[0]} to {vix_df.date.iloc[-1]}")
print(f"Latest VIX = {vix_df.close.iloc[-1]:.2f}")

# Get asset data
asset_data = {}
for sym in watchlist:
    df = get_stock_daily_data(sym, 500)
    asset_data[sym] = df
    print(f"{sym:15s}: {len(df)} days, close={df.close.iloc[-1]:.2f}")

# === Compute factor values over time ===
# We need to align dates. VIX data has its own date range.
# Compute 20d VIX rate of change on each date
vix_close = vix_df.set_index('date')['close']
vix_roc_20d = vix_close.pct_change(20).shift(0)  # 20-day ROC, no shift since we can compute on same date

# Get the set of common dates where we have both VIX and enough assets
common_dates = vix_roc_20d.dropna().index

# Compute factor values for each asset on each date
# Factor: when VIX is rising (vix_roc_20d > 0), favor safe havens
# When VIX is falling (vix_roc_20d < 0), favor risk assets
# Normalize the factor so it's comparable cross-sectionally

# Safe haven assets
safe_havens = ['XAU', 'US10Y', 'CN10Y']
risk_assets = [s for s in watchlist if s not in safe_havens]

print(f"\nSafe havens: {safe_havens}")
print(f"Risk assets: {risk_assets}")

# Build a date-aligned factor panel
factor_panel = []
ic_dates = []
ic_values = []

for i in range(60, len(common_dates)):
    current_date = common_dates[i]
    future_date_idx =