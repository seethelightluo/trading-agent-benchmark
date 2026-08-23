"""
Factor: cross_asset_dispersion_10d
Idea: Cross-sectional return dispersion across the 15-asset universe.
When dispersion is high, the strategy can differentiate between assets more effectively.
Construct: For each date, compute all 15 assets' 10-day forward returns,
then compute cross-sectional std of those returns. Normalize by rolling mean of dispersion.
Direction: +1 (high dispersion → more opportunity to pick winners)
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd
import numpy as np
from scipy import stats

# Get account and watchlist
acct = get_account_dict()
watchlist = acct['watch_list']
print(f"Watchlist: {watchlist}")
print(f"Number of assets: {len(watchlist)}")

# Fetch data for all assets - get enough history
data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=800)
    if df is not None and len(df) > 200:
        data[sym] = df
        print(f"{sym}: {len(df)} days, {df.date.iloc[0].strftime('%Y-%m-%d')} to {df.date.iloc[-1].strftime('%Y-%m-%d')}")
    else:
        print(f"{sym}: insufficient data ({len(df) if df is not None else 0})")

# Build aligned panel of close prices
prices = pd.DataFrame({sym: df.set_index('date')['close'] for sym, df in data.items()})
print(f"\nPrice panel shape: {prices.shape}")

# Sort by date
prices = prices.sort_index()
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

# Compute returns over different horizons
horizons = [1, 2, 3, 5, 10, 20]

# Factor: cross-sectional dispersion of 10-day returns
# For each date, compute the std of forward 10-day returns across assets
# Normalize by rolling mean of dispersion

def compute_dispersion(close_df, ret_window=10):
    """Compute cross-sectional return dispersion."""
    # Future returns
    fwd_returns = close_df.pct_change(ret_window).shift(-ret_window)
    
    # Daily dispersion (std across assets) of forward returns
    dispersion = fwd_returns.std(axis=1)
    
    # Normalize: z-score of dispersion vs rolling mean (60-day window)
    rolling_mean = dispersion.rolling(60, min_periods=20).mean()
    rolling_std = dispersion.rolling(60, min_periods=20).std()
    norm_dispersion = (dispersion - rolling_mean) / rolling_std.clip(lower=1e-8)
    
    return norm_dispersion, fwd_returns

norm_disp, fwd_ret_panel = compute_dispersion(prices, ret_window=10)

print(f"\nFactor (normalized dispersion) stats:")
print(f"  Non-null values: {norm_disp.notna().sum()}")
print(f"  Mean: {norm_disp.mean():.4f}")
print(f"  Std: {norm_disp.std():.4f}")
print(f"  Min: {norm_disp.min():.4f}")
print(f"  Max: {norm_disp.max():.4f}")

# Cross-sectional IC: correlate factor value with forward returns
# For each date, compute rank IC between disp_factor and forward returns
ic_values = {}
for h in horizons:
    fwd_ret = prices.pct_change(h).shift(-h)
    
    ics = []
    valid_dates = 0
    n_assets_used = []
    
    aligned_idx = norm_disp.dropna().index.intersection(fwd_ret.dropna(how='all').index)
    
    for dt in aligned_idx:
        f = norm_disp.loc[dt]
        r = fwd_ret.loc[dt]
        
        valid = r.notna() & ~np.isinf(r)
        n_valid = valid.sum()
        
        if n_valid >= 8:
            f_valid = f  # scalar, same for all assets
            r_valid = r[valid]
            
            # Rank IC: correlation between ranked factor (same value!) and ranked returns
            # Since factor is cross-sectional constant, we can't compute rank IC directly.
            # Instead, let's think differently...
            
    # Actually, the dispersion factor is a single value per date, not per asset.
    # This is a TIMING factor, not a cross-sectional ranking factor.
    # For a timing factor, we need to compute IC differently:
    # Predict if the next period will have high/low returns
    
    valid_dates_list.append(dt)
    n_assets_used.append(n_valid)

# Let me rethink - make it a cross-sectional factor that varies per asset
# Cross-sectional dispersion advantage: assets that have higher vol tend to 
# benefit more during high dispersion periods

print("\n\n=== REDEFINING: Making factor cross-sectional ===")
print("Factor: relative return dispersion sensitivity")
print("For each asset, compute: rolling correlation of asset return with cross-sectional dispersion")
print("Direction: +1 (assets that move with dispersion benefit from high-dispersion regimes)")

# Compute per-asset feature: rolling beta to cross-sectional dispersion
dispersion_raw = prices.pct_change(10).std(axis=1)  # daily cross-sectional std of 10-day returns

asset_factors = {}
for sym in prices.columns:
    ret_10d = prices[sym].pct_change(10)
    
    # Rolling correlation of asset return with dispersion
    roll_corr = ret_10d.rolling(60, min_periods=20).co