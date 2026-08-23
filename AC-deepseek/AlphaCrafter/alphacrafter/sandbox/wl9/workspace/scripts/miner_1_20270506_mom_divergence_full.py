"""
Explore: Momentum Divergence Factor (20d vs 60d) - Part 2 (validation logic)
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd
import numpy as np
from scipy import stats
import json
import os

CURRENT_DATE = "2027-05-06"
MIN_DAYS = 150

acc = get_account_dict()
watchlist = acc.get('watch_list', [])

data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=400)
    if df is not None and len(df) >= MIN_DAYS:
        data[sym] = df

print(f"Loaded {len(data)} instruments")

# Build aligned factor values
factor_values_dict = {}
forward_returns_dict = {}

for sym, df in data.items():
    df = df.copy()
    df['date_str'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)[:10])
    df['ret_20d'] = df['close'].shift(5) / df['close'].shift(25) - 1.0
    df['ret_60d'] = df['close'].shift(5) / df['close'].shift(65) - 1.0
    df['mom_div'] = df['ret_20d'] - df['ret_60d']
    df['fwd_10d'] = df['close'].shift(-10) / df['close'] - 1.0
    
    for _, row in df.iterrows():
        date_str = row['date_str']
        val = row['mom_div']
        fwd = row['fwd_10d']
        if not (np.isnan(val) or np.isnan(fwd)):
            if date_str not in factor_values_dict:
                factor_values_dict[date_str] = {}
                forward_returns_dict[date_str] = {}
            factor_values_dict[date_str][sym] = val
            forward_returns_dict[date_str][sym] = fwd

# Compute IC for each date
ic_values = []
valid_dates = []

for date_str in sorted(factor_values_dict.keys()):
    fvals = factor_values_dict[date_str]
    fretns = forward_returns_dict[date_str]
    
    # Only use symbols available in both
    common = [s for s in fvals if s in fretns]
    if len(common) >= 8:  # Minimum 8 instruments for valid cross-section
        x = np.array([fvals[s] for s in common])
        y = np.array([fretns[s] for s in common])
        
        # Check for constant values
        if np.std(x) > 1e-10 and np.std(y) > 1e-10:
            ic, _ = stats.spearmanr(x, y)
            ic_values.append(ic)
            valid_dates.append(date_str)

print(f"\n=== Momentum Divergence (20d vs 60d) Factor Validation ===")
print(f"Valid dates: {len(valid_dates)} (with >=8 instruments)")
print(f"Date range: {valid_dates[0]} to {valid_dates[-1]}" if valid_dates else "No valid dates")

if len(ic_values) > 0:
    ic_arr = np.array(ic_values)
    mean_ic = np.mean(ic_arr)
    std_ic = np.std(ic_arr)
    icir = mean_ic / std_ic if std_ic > 0 else 0
    hit_ratio = np.mean(ic_arr > 0)
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_arr))) if std_ic > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Mean IC (Spearman): {mean_ic:.6f}")
    print(f"  Std IC: {std_ic:.6f}")
    print(f"  ICIR: {icir:.6f}")
    print(f"  IC Hit Ratio (>0): {hit_ratio:.4f}")
    print(f"  T-stat: {t_stat:.4f}")
    print(f"  N dates: {len(ic_arr)}")
    
    # Decay analysis - forward returns at different horizons
    print(f"\n=== Decay Analysis ===")
    for horizon in [5, 10, 15, 20, 30]:
        ic_list = []
        for date_str in sorted(factor_values_dict.keys())[:-(horizon+10)]:
            fvals = factor_values_dict[date_str]
            fretns = forward_returns_dict[date_str]
            common = [s for s in fvals if s in fretns]
            if len(common) >= 8:
                x = np.array([fvals[s] for s in common])
                y = np.array([fretns[s] for s in common])
                if np.std(x) > 1e-10 and np.std(y) > 1e-10:
                    ic, _ = stats.spearmanr(x, y)
                    ic_list.append(ic)
        if len(ic_list) > 0:
            print(f"  Horizon {horizon}d: Mean IC={np.mean(ic_list):.6f}, N={len(ic_list)}")
    
    # Coverage
    total_sym_dates = sum(len(v) for v in factor_values_dict.values())
    covered_dates = len([d for d in factor_values_dict if len(factor_values_dict[d]) >= 8])
    total_dates = len(factor_values_dict)
    print(f"\nCoverage:")
    print(f"  Total date-symbol observations: {total_sym_dates}")
    print(f"  Dates with >=8 instruments: {covered_dates}/{total_dates} ({100*covered_dates/total_dates:.1f}%)")
    
    # Thresholds
    ic_pass = abs(mean_ic) >= 0.007
    icir_pass = abs(icir) >= 0.084
    
    print(f"\n=== Threshold Assessment ===")
    print(f"  |IC| >= 0.007: {ic_pass} (|IC|={abs(mean_ic):.6f})")
    print(f"  |ICIR| >= 0.084: {icir_pass} (|ICIR|={abs(icir):.6f})")
    
    if ic_pass and icir_pass:
        print("\n*** FACTOR PASSES - eligible for persistence ***")
    else:
        print("\n*** FACTOR FAILS - does not meet thresholds ***")
else:
    print("No valid IC observations")