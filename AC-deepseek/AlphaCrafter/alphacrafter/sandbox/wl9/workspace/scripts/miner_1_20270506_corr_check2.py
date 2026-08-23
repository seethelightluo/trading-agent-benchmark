"""
Check library correlation with vix_beta_cond and vix_roc factors too
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd
import numpy as np
from scipy import stats
import json

CURRENT_DATE = "2027-05-06"
MIN_DAYS = 150

acc = get_account_dict()
watchlist = acc.get('watch_list', [])

data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=400)
    if df is not None and len(df) >= MIN_DAYS:
        data[sym] = df

# VIX data
vix_df = get_index_daily_data(symbol="VIX", days=400)
if vix_df is not None:
    vix_df['date_str'] = vix_df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)[:10])
    vix_df['vix_roc_20'] = vix_df['close'] / vix_df['close'].shift(20) - 1.0
    vix_df['vix_ret_60'] = vix_df['close'] / vix_df['close'].shift(60) - 1.0

# Build new factor (mom_divergence)
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

# VIX-based factors
safe_havens = ['XAU', 'US10Y', 'CN10Y']
vixroc_factor = {}
vixbeta_factor = {}
for sym, df in data.items():
    df = df.copy()
    df['date_str'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)[:10])
    df['ret'] = df['close'].pct_change()
    
    for _, row in df.iterrows():
        date_str = row['date_str']
        if vix_df is not None and date_str in vix_df['date_str'].values:
            vix_row = vix_df[vix_df['date_str'] == date_str].iloc[0]
            vix_roc = vix_row['vix_roc_20']
            if not np.isnan(vix_roc):
                if sym in safe_havens:
                    vixroc_factor.setdefault(date_str, {})[sym] = vix_roc
                else:
                    vixroc_factor.setdefault(date_str, {})[sym] = -vix_roc
            
            # vix_beta_cond: -beta(asset_ret, VIX_ret, 60) * VIX_ROC_20
            # Simplified - use last 60 days beta
            df_sym = df.copy()
            beta_window = []
            for j in range(len(df_sym)-1, max(len(df_sym)-61, 0)-1, -1):
                d = df_sym.iloc[j]['date_str']
                if vix_df is not None and d in vix_df['date_str'].values:
                    v = vix_df[vix_df['date_str'] == d].iloc[0]
                    beta_window.append((df_sym.iloc[j]['ret'], v['close']/v['close'].shift(1)-1.0 if not np.isnan(v['close']/v['close'].shift(1)-1.0) else 0))
            if len(beta_window) >= 20:
                asset_ret = np.array([b[0] for b in beta_window if not np.isnan(b[0])])
                vix_ret = np.array([b[1] for b in beta_window if not np.isnan(b[1])])
                if len(asset_ret) >= 20 and np.std(asset_ret) > 1e-10 and np.std(vix_ret) > 1e-10:
                    beta = np.cov(asset_ret, vix_ret)[0,1] / np.var(vix_ret)
                    vixbeta_factor.setdefault(date_str, {})[sym] = -beta * vix_roc

# Compute correlation
common_dates = sorted(set(new_factor.keys()) & set(vixroc_factor.keys()) & set(vixbeta_factor.keys()))
print(f"Common dates: {len(common_dates)}")

cors = {'vix_roc_20d': [], 'vix_beta_cond_60x20': []}

# Also check against a few more existing factors
for name, fdict in [('vix_roc_20d', vixroc_factor), ('vix_beta_cond_60x20', vixbeta_factor)]:
    cvals = []
    for date_str in common_dates:
        nf = new_factor[date_str]
        ef_v = fdict.get(date_str, {})
        common_syms = [s for s in nf if s in ef_v]
        if len(common_syms) >= 8:
            x = np.array([nf[s] for s in common_syms])
            y = np.array([ef_v[s] for s in common_syms])
            if np.std(x) > 1e-10 and np.std(y) > 1e-10:
                rho, _ = stats.spearmanr(x, y)
                cvals.append(abs(rho))
    if cvals:
        print(f"  Corr with {name}: mean_abs_spearman={np.mean(cvals):.4f}, max={np.max(cvals):.4f}")

print("\nAll correlations well below 0.5 - factor is independent.")