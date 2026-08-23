"""
Check library correlation for mom_divergence against existing effective factors
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

# Compute our new factor: mom_divergence (20d - 60d skip5)
new_factor = {}
for sym, df in data.items():
    df = df.copy()
    df['date_str'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)[:10])
    df['ret_20d'] = df['close'].shift(5) / df['close'].shift(25) - 1.0
    df['ret_60d'] = df['close'].shift(5) / df['close'].shift(65) - 1.0
    df['mom_div'] = df['ret_20d'] - df['ret_60d']
    for _, row in df.iterrows():
        date_str = row['date_str']
        val = row['mom_div']
        if not np.isnan(val):
            if date_str not in new_factor:
                new_factor[date_str] = {}
            new_factor[date_str][sym] = val

# Compute existing factors for comparison
# 1. mom_120d_skip5: close.shift(5) / close.shift(125) - 1.0
# 2. mom_10d_skip5: close.shift(5) / close.shift(15) - 1.0
# 3. vix_beta_cond_60x20
# 4. vix_roc_20d

factors_to_check = {
    'mom_120d_skip5': lambda df: df['close'].shift(5) / df['close'].shift(125) - 1.0,
    'mom_10d_skip5': lambda df: df['close'].shift(5) / df['close'].shift(15) - 1.0,
}

def compute_factor_data(factor_func):
    fdict = {}
    for sym, df in data.items():
        df = df.copy()
        df['date_str'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)[:10])
        df['fval'] = factor_func(df)
        for _, row in df.iterrows():
            date_str = row['date_str']
            val = row['fval']
            if not (np.isnan(val) or np.isinf(val)):
                if date_str not in fdict:
                    fdict[date_str] = {}
                fdict[date_str][sym] = val
    return fdict

existing = {}
for name, func in factors_to_check.items():
    existing[name] = compute_factor_data(func)

# Compute cross-sectional correlations
common_dates = sorted(set(new_factor.keys()) & set(existing['mom_120d_skip5'].keys()) & set(existing['mom_10d_skip5'].keys()))
print(f"Common dates for correlation: {len(common_dates)}")

correlations = {}
for name in existing:
    cors = []
    for date_str in common_dates:
        nf = new_factor[date_str]
        ef = existing[name][date_str]
        common_syms = [s for s in nf if s in ef]
        if len(common_syms) >= 8:
            x = np.array([nf[s] for s in common_syms])
            y = np.array([ef[s] for s in common_syms])
            if np.std(x) > 1e-10 and np.std(y) > 1e-10:
                rho, _ = stats.spearmanr(x, y)
                cors.append(abs(rho))
    if cors:
        correlations[name] = np.mean(cors)
        print(f"  Corr with {name}: mean_abs_spearman={np.mean(cors):.4f}")

max_corr = max(correlations.values()) if correlations else 0
print(f"\nMax abs library correlation: {max_corr:.4f}")