#!/usr/bin/env python
"""
miner3_20320429_explore_factors.py
Current date: 2032-04-29

Explore novel factor ideas for the 15 cross-asset universe.
All existing factors were evicted due to pairwise correlation conflicts with usdcny_beta_60 (deprecated).
Goal: find factors with IC >= 0.0070, ICIR >= 0.0840, and low mutual correlation (< 0.5).

Idea 1: Cross-sectional volatility dispersion (xs_vol_dispersion_20)
  - When dispersion across assets is high, momentum/trend works better
  - When dispersion is low, uniform risk-on/risk-off dominates
  - Signal: std of 20d returns across assets (lower = uniform, higher = divergent)
  
Idea 2: US10Y momentum adjusted for regime (us10y_mom_regime_20)
  - US10Y yield change as macro signal
  - Rising yields: negative for equities, mixed for commodities
  - Keep it simple: use 20d yield change

Idea 3: Commodity momentum relative to gold (commodity_vs_gold_mom_20)
  - Commodities (WTI + COPPER) momentum minus gold (XAU) momentum
  - When commodities > gold: growth/inflation regime
  - When gold > commodities: risk-off/defensive regime

Idea 4: Composite safe-haven score (safe_haven_score_20)
  - Weighted combination: gold momentum - equity momentum + yield decline momentum
  - Higher values = risk-off preference

Idea 5: Short-term reversal (st_reversal_5) 
  - 5-day reversal: negative of 5-day return
  - Contrarian signal in high-volatility regimes
"""

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

# Get watchlist
acc = get_account_dict()
watch_list = acc['watch_list']
print(f"Watchlist ({len(watch_list)}): {watch_list}")

# Fetch data
LOOKBACK = 800
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

# Also fetch observation-only data for macro signals
macro_symbols = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
for symbol in macro_symbols:
    df = get_index_daily_data(symbol=symbol, days=LOOKBACK)
    if df is not None and len(df) >= 60:
        data[symbol] = df
        print(f"{symbol} (macro): {len(df)} days [{df['date'].iloc[0]}..{df['date'].iloc[-1]}]")

# Build aligned price matrix
all_dates = set()
for sym, df in data.items():
    all_dates.update(pd.to_datetime(df['date']))
all_dates = sorted(all_dates)
print(f"\nTotal trading days: {len(all_dates)}")

close_df = pd.DataFrame(index=all_dates, columns=list(data.keys()))
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
ret_1d = close_df.pct_change()

# Enable using macro-only instruments for factor computation (they're observation only for orders)
tradable = watch_list.copy()
all_syms = list(data.keys())

def compute_ic_series(factor_values, forward_returns, min_valid=8):
    """Compute daily cross-sectional IC between factor and forward returns."""
    ic_vals = []
    n_valid_list = []
    dates_used = []
    
    for date in factor_values.index:
        if date not in forward_returns.index:
            continue
        fv = factor_values.loc[date]
        fr = forward_returns.loc[date]
        
        # Only consider tradable assets
        valid_mask = fv.notna() & fr.notna()
        valid_fv = fv[valid_mask]
        valid_fr = fr[valid_mask]
        
        # Only assets in tradable list
        valid_fv = valid_fv[[s for s in valid_fv.index if s in tradable]]
        valid_fr = valid_fr[[s for s in valid_fr.index if s in tradable]]
        
        valid_fv = valid_fv.dropna()
        valid_fr = valid_fr.dropna()
        
        common = valid_fv.index.intersection(valid_fr.index)
        if len(common) >= min_valid:
            ic, _ = spearmanr(valid_fv[common], valid_fr[common])
            if not np.isnan(ic):
                ic_vals.append(ic)
                n_valid_list.append(len(common))
                dates_used.append(date)
    
    return np.array(ic_vals), np.array(n_valid_list), pd.DatetimeIndex(dates_used)