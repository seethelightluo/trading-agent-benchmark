"""
miner3_20270826_exp01: Cross-sectional relative momentum
Measures each asset's recent return relative to the cross-sectional median.
This captures outperformance/underperformance vs the cross-asset "market"
which differs from absolute momentum (mom_10d, mom_120d in the ensemble).
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

acc = get_account_dict()
watch_list = acc.get('watch_list', [])
print(f"Watchlist: {watch_list}")
print(f"Assets: {len(watch_list)}")

# Get enough data
N_DAYS = 1500
data = {}
for sym in watch_list:
    df = get_stock_daily_data(sym, days=N_DAYS)
    if df is not None and len(df) >= 500:
        df['date'] = pd.to_datetime(df['date'])
        data[sym] = df
    else:
        print(f"Skipping {sym} - insufficient data")

print(f"Loaded {len(data)} assets")

# Build a panel of close prices
closes = {}
for sym, df in data.items():
    cl = df[['date', 'close']].set_index('date')['close']
    closes[sym] = cl

# Build close DataFrame on union of dates
all_dates = sorted(set().union(*[set(cl.index) for cl in closes.values()]))
close_df = pd.DataFrame(closes, index=all_dates)
close_df = close_df.ffill()
print(f"Total dates: {len(close_df)}, Date range: {close_df.index[0].strftime('%Y-%m-%d')} to {close_df.index[-1].strftime('%Y-%m-%d')}")

# Compute returns for various horizons
horizons = [5, 10, 21, 63]
forward_horizon = 10

results = {}
for lookback in horizons:
    rets = close_df.pct_change(lookback)
    fwd_rets = close_df.pct_change(forward_horizon).shift(-forward_horizon)
    
    # Cross-sectional median for each date
    med_rets = rets.median(axis=1)
    
    # Factor: asset return minus cross-sectional median
    factor = rets.subtract(med_rets, axis=0)
    
    # Compute IC across dates
    ic_list = []
    for i in range(lookback, len(factor) - forward_horizon):
        dt = factor.index[i]
        f_vals = factor.iloc[i].dropna()
        r_vals = fwd_rets.iloc[i].dropna()
        common = f_vals.index.intersection(r_vals.index)
        if len(common) >= 8:
            f_s = f_vals[common].values
            r_s = r_vals[common].values
            if np.std(f_s) > 0 and np.std(r_s) > 0:
                ic = np.corrcoef(f_s, r_s)[0,1]
                ic_list.append(ic)
    
    n_ic_dates = len(ic_list)
    if n_ic_dates > 0:
        mean_ic = float(np.mean(ic_list))
        std_ic = float(np.std(ic_list))
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        hit_ratio = float(np.mean([1 if ic > 0 else 0 for ic in ic_list]))
        
        # Decay analysis at horizons 1,2,3,5,10,20
        decay_ic = {}
        for dh in [1, 2, 3, 5, 10, 20]:
            dh_fwd = close_df.pct_change(dh).shift(-dh)
            dh_ic = []
            for i in range(lookback, len(factor) - dh):
                f_vals = factor.iloc[i].dropna()
                r_vals = dh_fwd.iloc[i].dropna()
                common = f_vals.index.intersection(r_vals.index)
                if len(common) >= 8:
                    f_s = f_vals[common].values
                    r_s = r_vals[common].values
                    if np.std(f_s) > 0 and np.std(r_s) > 0:
                        dh_ic.append(np.corrcoef(f_s, r_s)[0,1])
            decay_ic[str(dh)] = float(np.mean(dh_ic) if dh_ic else 0.0)
        
        # Coverage
        valid_mask = factor.notna()
        coverage = float(valid_mask.mean().mean())
        
        # Turnover: rank correlation between consecutive factor values
        turnover_list = []
        for i in range(1, len(factor)):
            prev = factor.iloc[i-1].dropna()
            curr = factor.iloc[i].dropna()
            common = prev.index.intersection(curr.index)
            if len(common) >= 8:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(prev[common], curr[common])
                turnover_list.append(1 - rho)
        turnover = float(np.mean(turnover_list)) if turnover_list else 0.5
        
        results[lookback] = {
            'factor_name': f'cs_rel_mom_{lookback}d',
            'lookback': lookback,
            'n_ic_dates': n_ic_dates,
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'icir': icir,
            'hit_ratio': hit_ratio,
            'coverage': coverage,
            'turnover': turnover,
            'decay_ic': decay_ic
        }
        print(f"\n--- CS Relative Momentum {lookback}d ---")
        print(f"  n_dates: {n_ic_dates}")
        print(f"  IC: {mean_ic:.6f}")
        print(f"  IC std: {std_ic:.6f}")
        print(f"  ICIR: {icir:.6f}")
        print(f"  Hit ratio: {hit_ratio:.4f}")
        print(f"  Coverage: {coverage:.4f}")
        print(f"  Turnover: {turnover:.4f}")
        print(f"  Decay: {decay_ic}")
        
        # Check threshold
        passes_ic = abs(mean_ic) >= 0.007
        passes_icir = abs(icir) >= 0.084
        print(f"  Passes IC>=0.007? {passes_ic}, Passes ICIR>=0.084? {passes_icir}")
    else:
        print(f"\n--- CS Relative Momentum {lookback}d: No valid IC dates ---")

# Print summary
print("\n\n=== SUMMARY ===")
for lb, r in sorted(results.items()):
    gate = "PASS" if (abs(r['mean_ic'])>=0.007 and abs(r['icir'])>=0.084) else "FAIL"
    print(f"{r['factor_name']}: IC={r['mean_ic']:.4f}, ICIR={r['icir']:.4f}, Hit={r['hit_ratio']:.4f} -> {gate}")