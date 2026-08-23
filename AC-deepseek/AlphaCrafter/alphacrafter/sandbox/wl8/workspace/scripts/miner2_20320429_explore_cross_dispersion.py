"""
Cross-Asset Dispersion & Relative Resilience Factor (miner2, 2032-04-29)

Idea: In a high-volatility divergent regime (VIX ~34, SPX -8.1%, BTC -16.9%),
assets that exhibit 'relative resilience' - showing less drawdown than the 
cross-sectional median during stress periods - may predict mean-reversion 
or continued resilience.

Construction: For each asset, compute the ratio of its 10-day return to the 
cross-sectional median absolute 10-day return. Assets that outperform the 
median (higher ratio = less damaged) are expected to either continue 
outperforming or revert. We test both directions.

Factor definition: rank_cross_return = rank(pct_change_10d) / n_assets
Then compute: raw_factor = rank_cross_return - median(rank_cross_return)
This normalizes across different market regimes.

We also test a "divergence capture" variant: abs_dispersion_20 = 
std(cross_sectional_returns over 20d) / mean(abs(cross_sectional_returns))
Large dispersion periods often precede mean reversion.
"""

import sys
sys.path.insert(0, '.')
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import numpy as np
import pandas as pd
from datetime import datetime

CURRENT_DATE = "2032-04-29"
HISTORY_DAYS = 500  # sufficient window

# Universe: 15 tradable assets
WATCH_LIST = [
    "000300.SH", "SPX", "HSI", "N225", "SX5E",
    "000688.SH", "SOX", "NDX",
    "XAU", "COPPER", "WTI",
    "BTC", "ETH",
    "US10Y", "CN10Y"
]

def fetch_data(symbol, days=HISTORY_DAYS):
    """Fetch data for a symbol, trying stock first then index."""
    df = get_stock_daily_data(symbol, days)
    if df is None or len(df) < 30:
        df = get_index_daily_data(symbol, days)
    return df

def compute_cross_dispersion_factor(price_dict, lookback=10):
    """
    Factor: cross-asset relative resilience.
    
    For each date with sufficient data:
    1. Compute 10-day returns for all assets
    2. Rank assets by return
    3. Factor = rank - 0.5 (centered around zero)
    4. Higher = outperformer (long these), Lower = underperformer
    """
    aligned = None
    for sym, df in price_dict.items():
        if df is None or len(df) < lookback + 5:
            continue
        close = df['close'].copy()
        close.name = sym
        if aligned is None:
            aligned = close.to_frame()
        else:
            aligned = aligned.join(close, how='outer')
    
    if aligned is None or aligned.shape[1] < 8:
        return None, None
    
    # Compute returns
    rets = aligned.pct_change(lookback)
    
    # Cross-dispersion: std of returns across assets
    cross_std = rets.std(axis=1)
    cross_mean_abs = rets.abs().mean(axis=1)
    dispersion = cross_std / cross_mean_abs.where(cross_mean_abs > 1e-6, 1e-6)
    
    # Rank-based factor: for each date, rank assets by forward-looking return
    # To avoid lookahead, we use rank of asset returns (which is known)
    # Factor value = rank of return, centered and scaled
    factor_df = rets.rank(axis=1, pct=True) - 0.5
    
    return factor_df, dispersion

def compute_ic(factor_df, forward_rets, n_min=8):
    """
    Compute cross-sectional IC between factor and forward returns.
    Returns a series of IC values per date.
    """
    ics = []
    dates_used = 0
    for date in factor_df.index:
        if date not in forward_rets.index:
            continue
        f = factor_df.loc[date].dropna()
        r = forward_rets.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_min:
            continue
        fv = f[common].values
        rv = r[common].values
        if np.std(fv) < 1e-10 or np.std(rv) < 1e-10:
            continue
        corr = np.corrcoef(fv, rv)[0, 1]
        if not np.isnan(corr):
            ics.append(corr)
            dates_used += 1
    return np.array(ics), dates_used

print(f"=== Cross-Dispersion Factor Exploration ({CURRENT_DATE}) ===")
print(f"Universe: {len(WATCH_LIST)} assets")

# Fetch data
price_dict = {}
for sym in WATCH_LIST:
    df = fetch_data(sym)
    if df is not None and len(df) > 50:
        price_dict[sym] = df
        print(f"  {sym}: {len(df)} days available")
    else:
        print(f"  {sym}: insufficient data, skipping")

print(f"\nAssets with data: {len(price_dict)}")

# Compute factor
factor_df, dispersion = compute_cross_dispersion_factor(price_dict, lookback=10)

if factor_df is not None:
    print(f"\nFactor shape: {factor_df.shape}")
    print(f"Factor date range: {factor_df.index[0]} to {factor_df.index[-1]}")
    
    # Get forward 5-day returns for IC calculation
    # Build aligned close prices
    aligned = None
    for sym, df in price_dict.items():
        close = df['close'].copy()
        close.name = sym
        if aligned is None:
            aligned = close.to_frame()
        else:
            aligned = aligned.join(close, how='outer')
    aligned = aligned.dropna(how='all')
    
    # Compute forward returns for different horizons
    for fwd_horizon in [5, 10, 20]:
        fwd_rets = aligned.pct_change(fwd_horizon).shift(-fwd_horizon)
        
        # Align factor and forward returns
        common_dates = factor_df.index.intersection(fwd_rets.index)
        f_aligned = factor_df.loc[common_dates]
        r_aligned = fwd_rets.loc[common_d