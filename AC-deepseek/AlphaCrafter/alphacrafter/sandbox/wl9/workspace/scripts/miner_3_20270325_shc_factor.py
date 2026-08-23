"""
Factor: safe_haven_composite_20d (SHC) - v2
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

acct = get_account_dict()
watchlist = acct['watch_list']
print(f"Watchlist ({len(watchlist)}): {watchlist}")

# Price data
prices = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=500)
    if df is not None and len(df) > 100:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        prices[sym] = df['close']

price_df = pd.DataFrame(prices).sort_index()
print(f"Price panel: {price_df.shape}, {price_df.index[0].strftime('%Y-%m-%d')} to {price_df.index[-1].strftime('%Y-%m-%d')}")

# VIX
vix_df = get_index_daily_data(symbol='VIX', days=500)
vix = None
if vix_df is not None and len(vix_df) > 100:
    vix = vix_df.set_index('date')['close']
    print(f"VIX: {len(vix_df)} days")
else:
    vix_df = get_stock_daily_data(symbol='VIX', days=500)
    if vix_df is not None and len(vix_df) > 100:
        vix = vix_df.set_index('date')['close']
        print(f"VIX stock: {len(vix_df)} days")

# Returns
returns = price_df.pct_change()

# 1. Negative vol z-score
w = 20
rw = 60
rolling_vol = returns.rolling(w).std()
vol_mean = rolling_vol.rolling(rw, min_periods=30).mean()
vol_std = rolling_vol.rolling(rw, min_periods=30).std()
vol_z = (rolling_vol - vol_mean) / vol_std.clip(lower=1e-8)
vol_component = -vol_z

# 2. VIX correlation negated
if vix is not None:
    vix_ret = vix.pct_change()
    common_idx = returns.index.intersection(vix_ret.dropna().index)
    ret_aligned = returns.loc[common_idx]
    vix_ret_aligned = vix_ret.loc[common_idx]
    vix_corr = ret_aligned.rolling(40, min_periods=20).corr(vix_ret_aligned)
    vix_component = -vix_corr
else:
    vix_component = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

# 3. SMA ratio
sma20 = price_df.rolling(w).mean()
sma_ratio = price_df / sma20 - 1.0

# Composite
factor = (0.5 * vol_component + 0.3 * vix_component + 0.2 * sma_ratio).replace([np.inf, -np.inf], np.nan)

# Coverate check
n_valid_total = factor.notna().sum().sum()
print(f"\nTotal valid values: {n_valid_total}")
print(f"Coverage per asset: {(factor.notna().sum() / len(factor)).values}")

# Horizon IC analysis
print("\n=== IC Analysis ===")
for h in [1, 2, 3, 5, 10, 20]:
    fwd_ret = price_df.pct_change(h).shift(-h)
    common_idx = factor.dropna(how='all').index.intersection(fwd_ret.dropna(how='all').index)
    
    ics = []
    n_assets_per_date = []
    for dt in common_idx:
        f = factor.loc[dt]
        r = fwd_ret.loc[dt]
        mask = r.notna() & f.notna() & ~np.isinf(r) & ~np.isinf(f)
        if mask.sum() >= 8:
            f_vals = f[mask]; r_vals = r[mask]
            ic, _ = scipy_stats.spearmanr(f_vals.rank(), r_vals.rank())
            ics.append(ic)
            n_assets_per_date.append(mask.sum())
    
    if len(ics) > 0:
        avg_ic = np.mean(ics)
        std_ic = np.std(ics) if len(ics) > 1 else 0.001
        icir = avg_ic / std_ic if std_ic > 0 else 0.0
        hit = np.mean([1 for ic in ics if ic > 0])
        
        print(f"Horizon {h:2d}d: {len(ics):4d} dates, IC={avg_ic:.4f}, ICIR={icir:.4f}, Hit={hit:.4f}, AvgAssets={np.mean(n_assets_per_date):.1f}")
        
        if h == 10:
            print(f"  >> Admission: |IC|={abs(avg_ic):.4f} >= 0.007 ? {abs(avg_ic) >= 0.007}")
            print(f"  >> Admission: |ICIR|={abs(icir):.4f} >= 0.084 ? {abs(icir) >= 0.084}")