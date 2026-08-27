"""Explore current market state for factor mining context"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd
import numpy as np

watchlist = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

# Get account state
acct = get_account_dict()
print(f"Account net_assets: {acct.get('net_assets', 'N/A')}")

# Get recent 500 days data to understand current market regime
data = {}
for sym in watchlist:
    df = get_stock_daily_data(sym, 500)
    if df is not None and len(df) > 0:
        data[sym] = df
        print(f"{sym}: {len(df)} days, last close={df['close'].iloc[-1]:.2f}, date_range={df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    else:
        print(f"{sym}: NO DATA")

# Compute returns for last 60 days
print("\n\n=== Last 60d cumulative returns ===")
for sym in watchlist:
    if sym in data and len(data[sym]) >= 60:
        df = data[sym]
        ret_60 = df['close'].iloc[-1] / df['close'].iloc[-60] - 1
        ret_20 = df['close'].iloc[-1] / df['close'].iloc[-20] - 1
        ret_5 = df['close'].iloc[-1] / df['close'].iloc[-5] - 1
        vol_20 = df['pct_change'].iloc[-20:].std() * np.sqrt(252)
        print(f"{sym:15s}  ret_5d={ret_5*100:6.2f}%  ret_20d={ret_20*100:6.2f}%  ret_60d={ret_60*100:6.2f}%  vol_20d={vol_20*100:5.2f}%")

# Also check macro signals
print("\n\n=== Macro Signals (DXY, VIX, etc) ===")
for sym in ["DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"]:
    df_idx = get_index_daily_data(sym, 200)
    if df_idx is not None and len(df_idx) > 0:
        last_close = df_idx['close'].iloc[-1]
        ret_60 = df_idx['close'].iloc[-1] / df_idx['close'].iloc[-60] - 1 if len(df_idx) >= 60 else 0
        print(f"{sym:15s}  last_close={last_close:.2f}  ret_60d={ret_60*100:.2f}%  n_obs={len(df_idx)}")

print("\nDone.")
