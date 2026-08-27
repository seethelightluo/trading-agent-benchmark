"""
Idea 1: DXY_FWD_BETA - Beta of assets with respect to DXY (trade-weighted dollar index)
Rationale: DXY is falling (-4.5% in 60d, at 95.90). Assets with high negative DXY beta benefit
from USD weakness. This complements the existing beta_VIX_60 factor.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

watchlist = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

# Get DXY data
dxy = get_index_daily_data("DXY", 500)
if dxy is None or len(dxy) < 120:
    print("Cannot get DXY data")
    exit()

# For each asset, compute rolling 60d beta to DXY returns
results = []
for sym in watchlist:
    df = get_stock_daily_data(sym, 500)
    if df is None or len(df) < 120:
        continue
    
    # Align dates
    df = df[['date','close','pct_change']].copy()
    dxy_aligned = dxy[['date','close','pct_change']].copy()
    dxy_aligned = dxy_aligned.rename(columns={'pct_change':'dxy_ret','close':'dxy_close'})
    
    merged = pd.merge(df, dxy_aligned, on='date', how='inner').dropna()
    if len(merged) < 120:
        continue
    
    # Rolling 60d DXY beta
    merged['dxy_beta_60'] = merged['pct_change'].rolling(60).cov(merged['dxy_ret']) / merged['dxy_ret'].rolling(60).var()
    merged['dxy_beta_20'] = merged['pct_change'].rolling(20).cov(merged['dxy_ret']) / merged['dxy_ret'].rolling(20).var()
    
    latest = merged.iloc[-1]
    results.append({
        'symbol': sym,
        'dxy_beta_60': latest['dxy_beta_60'],
        'dxy_beta_20': latest['dxy_beta_20'],
        'dxy_close': latest['dxy_close'],
        'last_close': latest['close'],
        'n_obs': len(merged)
    })
    print(f"{sym:15s} dxy_beta_60={latest['dxy_beta_60']:+.4f} dxy_beta_20={latest['dxy_beta_20']:+.4f} n={len(merged)}")

# Now validate predictive power: compute IC of DXY beta vs forward returns
print("\n\n=== DXY Beta Predictive Validation ===")
print("Testing if assets with high negative DXY beta outperform when DXY falls")

all_ics = []
for test_days in [5, 10, 20]:
    ics = []
    for i in range(120, len(merged) - test_days):
        # Get cross-section at this date
        cs_values = []
        cs_fwd_ret = []
        for sym in watchlist:
            df = get_stock_daily_data(sym, 500)
            if df is None or len(df) < i + test_days + 1:
                continue
            row = df.iloc[i]
            fwd_row = df.iloc[i + test_days]
            prev_row = df.iloc[i-1]  # for pct_change proxy
            
            # Compute DXY beta at this point
            slice_df = df.iloc[max(0,i-59):i+1]
            dxy_slice = dxy.iloc[max(0,i-59):i+1]
            if len(slice_df) < 20 or len(dxy_slice) < 20:
                continue
            
            # Align them
            sdf = pd.DataFrame({'ret': slice_df['pct_change'].values, 'dxy_ret': dxy_slice['pct_change'].values}).dropna()
            if len(sdf) < 20:
                continue
            beta = sdf['ret'].cov(sdf['dxy_ret']) / sdf['dxy_ret'].var()
            
            fwd_ret = fwd_row['close'] / row['close'] - 1
            cs_values.append(beta)
            cs_fwd_ret.append(fwd_ret)
        
        if len(cs_values) >= 8:
            corr = np.corrcoef(cs_values, cs_fwd_ret)[0,1]
            ics.append(corr)
    
    if len(ics) > 0:
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        icir = mean_ic / std_ic * np.sqrt(len(ics)) if std_ic > 0 else 0
        hit_rate = np.mean(np.array(ics) > 0)
        all_ics.append({'horizon': test_days, 'mean_ic': mean_ic, 'std_ic': std_ic, 'icir': icir, 'hit_rate': hit_rate, 'n_dates': len(ics)})
        print(f"Fwd {test_days}d: mean_IC={mean_ic:.6f} std_IC={std_ic:.6f} ICIR={icir:.6f} hit={hit_rate:.2%} n_dates={len(ics)}")

# Also test with sign-flipped (since DXY negative beta is desirable when DXY falls)
print("\n\n=== Negative DXY Beta (High = good when DXY falling) ===")
for entry in all_ics:
    print(f"Fwd {entry['horizon']}d: mean_IC={entry['mean_ic']:.6f} ICIR={entry['icir']:.6f} (using raw DXY beta)")
    
print("\n\nNote: If IC is negative, that means assets with HIGH DXY beta (positive) underperform when DXY falls,")
print("so flipping sign makes it: -dxy_beta = DXY-hedge factor")
print("Gate: abs(IC) >= 0.0070 and abs(ICIR) >= 0.0840")
