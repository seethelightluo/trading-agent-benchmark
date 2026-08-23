#!/usr/bin/env python
"""
miner3_20320401_novel_factor_exploration.py
Current date: 2032-04-01

Explore novel factor ideas for the 15 cross-asset universe.
All existing factors were evicted due to pairwise correlation conflicts > 0.5.
Goal: find factors with IC >= 0.0070, ICIR >= 0.0840, AND low correlation (< 0.5)
with the three fallback ensemble factors (mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20).

Idea 1: Cross-Asset Momentum Divergence (crypto_equity_div_20)
- Measures relative strength of crypto (BTC+ETH) vs equities (SPX+NDX+HSI) 
- Higher values = crypto outperforming equities (risk-on rotation preference)
- This is a relative-strength signal, not a momentum signal per se

Idea 2: HV-to-RV Ratio (hvrv_ratio_20)
- 20d historical volatility / (sum of absolute daily returns / sum of daily close-to-close range)
- Captures volatility efficiency - when HV is high relative to RV, suggests directional moves vs noise

Idea 3: Commodity Momentum Decorrelated (xau_wti_mom_20)
- Average of XAU and WTI 20d momentum minus cross-sectional average momentum
- Measures commodity-specific relative strength net of broad market

Let's test these ideas.
"""

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

# Get watchlist
acc = get_account_dict()
watch_list = acc['watch_list']
print(f"Watchlist ({len(watch_list)}): {watch_list}")

# Fetch data - use max available (500 trading days ~ 2 years)
LOOKBACK = 500
data = {}
for symbol in watch_list:
    df = get_stock_daily_data(symbol=symbol, days=LOOKBACK)
    if df is None or len(df) < 60:
        df = get_index_daily_data(symbol=symbol, days=LOOKBACK)
    if df is not None and len(df) >= 60:
        data[symbol] = df
        print(f"{symbol}: {len(df)} days [{df['date'].iloc[0]}..{df['date'].iloc[-1]}]")
    else:
        print(f"{symbol}: insufficient data")

# Ensure uniform date indexing
all_dates = set()
for sym, df in data.items():
    all_dates.update(pd.to_datetime(df['date']))
all_dates = sorted(all_dates)
print(f"\nTotal trading days: {len(all_dates)}")

# Build aligned price matrix
close_df = pd.DataFrame(index=all_dates, columns=data.keys())
for sym, df in data.items():
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    close_df[sym] = df['close']

close_df = close_df.dropna(how='all')
print(f"Close matrix shape: {close_df.shape}")
print(f"Date range: {close_df.index[0]} to {close_df.index[-1]}")

# Compute returns
ret_df = close_df.pct_change()

# ============================================================
# FACTOR 1: Crypto-Equity Divergence (crypto_equity_div_20)
# ============================================================
print("\n\n=== FACTOR 1: Crypto-Equity Divergence (crypto_equity_div_20) ===")
print("Rationale: measures relative momentum of crypto vs equities over 20d.")

def compute_crypto_equity_div(close, window=20):
    """Factor: crypto_equity_div_20
    crypto_avg_ret_20d - equity_avg_ret_20d
    High values = crypto outperforming equities (risk-on rotation)
    """
    ret = close.pct_change(window)
    crypto_list = ['BTC', 'ETH']
    equity_list = ['SPX', 'NDX', 'HSI', '000300.SH', 'N225', 'SX5E']
    
    crypto_ret = ret[crypto_list].mean(axis=1, skipna=False)
    equity_ret = ret[equity_list].mean(axis=1, skipna=False)
    
    factor = crypto_ret - equity_ret
    return factor

# Compute factor 1 values
f1_values = compute_crypto_equity_div(close_df, window=20)
f1_name = "crypto_equity_div_20"
print(f"{f1_name} computed, shape={f1_values.shape}, date range={f1_values.index[0]}..{f1_values.index[-1]}")

# ============================================================
# FACTOR 2: HV/RV Ratio (hvrv_ratio_20) 
# ============================================================
print("\n\n=== FACTOR 2: HV/RV Ratio (hvrv_ratio_20) ===")
print("Rationale: Historical volatility relative to realized range-based vol.")
print("High values suggest directional runs, low values suggest noise.")

def compute_hvrv_ratio(close, window