"""
Script: miner_3_20270617_rel_momentum.py
Explores: Cross-asset Relative Momentum (relative strength vs cross-asset median)
Purpose: Each asset's cumulative return minus the cross-sectional median return 
         over a lookback window. This isolates idiosyncratic momentum from 
         common market moves.
"""

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

WATCHLIST = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E', 'SOX', 'NDX', 
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MIN_LOOKBACK = 260  # ~1 year
FORWARD_HORIZONS = [1, 2, 3, 5, 10, 20]

print("=" * 70)
print("Cross-Asset Relative Momentum (Relative Strength vs Median)")
print("=" * 70)

# Fetch data
data = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=2000)
    if df is not None and len(df) > MIN_LOOKBACK:
        df['date'] = pd.to_datetime(df['date'])
        data[sym] = df

# Align dates
all_dates = None
for sym, df in data.items():
    if all_dates is None:
        all_dates = set(df['date'].values)
    else:
        all_dates = all_dates.intersection(set(df['date'].values))
all_dates = sorted(all_dates)

print(f"\nAssets with data: {len(data)}")
print(f"Common trading days: {len(all_dates)}")

# Build aligned close matrix
close_df = pd.DataFrame(index=all_dates)
for sym, df in data.items():
    df_idx = df.set_index('date')
    close_df[sym] = df_idx.loc[all_dates, 'close']

print(f"\nClose matrix shape: {close_df.shape}")
print(f"Date range: {close_df.index[0]} to {close_df.index[-1]}")

# Test various lookback windows
for lookback in [20, 40, 60, 120]:
    print(f"\n--- Lookback: {lookback}d ---")
    
    for horizon in FORWARD_HORIZONS:
        if horizon > lookback:
            continue
            
        # Compute cumulative returns over lookback
        cum_ret = close_df.pct_change(lookback)
        
        # Compute cross-sectional median of cumulative returns
        cs_median = cum_ret.median(axis=1)
        
        # Relative momentum = asset return - cross-section median
        rel_mom = cum_ret.subtract(cs_median, axis=0)
        
        # Forward returns
        fwd_ret = close_df.pct_change(horizon).shift(-horizon)
        
        # Align
        valid_idx = rel_mom.dropna(how='all').index.intersection(fwd_ret.dropna(how='all').index)
        if len(valid_idx) < 6:
            continue
            
        rel_mom_v = rel_mom.loc[valid_idx]
        fwd_ret_v = fwd_ret.loc[valid_idx]
        
        # Compute rank IC
        ic_list = []
        n_assets_avg = 0
        for t in valid_idx:
            row_panel = rel_mom_v.loc[t].dropna()
            fwd_panel = fwd_ret_v.loc[t].dropna()
            common_syms = row_panel.index.intersection(fwd_panel.index)
            if len(common_syms) >= 8:
                rm = row_panel[common_syms].rank()
                fr = fwd_panel[common_syms].rank()
                ic, _ = spearmanr(rm, fr)
                if not np.isnan(ic):
                    ic_list.append(ic)
                n_assets_avg += len(common_syms)
        
        if len(ic_list) < 5:
            continue
            
        mean_ic = np.mean(ic_list)
        std_ic = np.std(ic_list)
        icir = mean_ic / std_ic * np.sqrt(len(ic_list)) if std_ic > 0 else 0
        hit_ratio = np.mean(np.array(ic_list) > 0)
        
        print(f"  Horizon {horizon}d: IC={mean_ic:.5f}, ICIR={icir:.6f}, Hit={hit_ratio:.3f}, "
              f"N_obs={len(ic_list)}, Avg_assets={n_assets_avg/len(ic_list):.1f}")

print("\n\n=== Decay Analysis (lookback=40d) ===")
for horizon in [1, 2, 3, 5, 10, 20]:
    cum_ret = close_df.pct_change(40)
    cs_median = cum_ret.median(axis=1)
    rel_mom = cum_ret.subtract(cs_median, axis=0)
    
    fwd_ret = close_df.pct_change(horizon).shift(-horizon)
    
    valid_idx = rel_mom.dropna(how='all').index.intersection(fwd_ret.dropna(how='all').index)
    if len(valid_idx) < 6:
        continue
        
    rel_mom_v = rel_mom.loc[valid_idx]
    fwd_ret_v = fwd_ret.loc[valid_idx]
    
    ic_list = []
    fo