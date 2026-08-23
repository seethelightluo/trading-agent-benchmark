"""Check how many dates have ge8 valid instruments and look at coverage"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

acct = get_account_dict()
watchlist = acct['watch_list']

prices = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=500)
    if df is not None and len(df) > 100:
        prices[sym] = df.set_index('date')['close']
price_df = pd.DataFrame(prices).sort_index()

returns = price_df.pct_change()

# VIX
vix_df = get_index_daily_data(symbol='VIX', days=500)
vix = vix_df.set_index('date')['close'] if vix_df is not None and len(vix_df) > 100 else None
vix_ret = vix.pct_change() if vix is not None else None

w, rw = 20, 60
rolling_vol = returns.rolling(w).std()
vol_mean = rolling_vol.rolling(rw, min_periods=30).mean()
vol_std = rolling_vol.rolling(rw, min_periods=30).std()
vol_z = (rolling_vol - vol_mean) / vol_std.clip(lower=1e-8)
vol_component = -vol_z

if vix is not None:
    common = returns.index.intersection(vix_ret.dropna().index)
    ret_a = returns.loc[common]; vix_a = vix_ret.loc[common]
    vix_corr = ret_a.rolling(40, min_periods=20).corr(vix_a)
    vix_comp = -vix_corr
else:
    vix_comp = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

sma20 = price_df.rolling(w).mean()
sma_ratio = price_df / sma20 - 1.0

factor = (0.5 * vol_component + 0.3 * vix_comp + 0.2 * sma_ratio).replace([np.inf, -np.inf], np.nan)

# Coverage: dates with ge8 valid
factor_valid = factor.notna().sum(axis=1)
dates_ge8 = (factor_valid >= 8).sum()
total_dates = len(factor_valid)
print(f"Dates with >=8 valid: {dates_ge8}/{total_dates} ({dates_ge8/total_dates*100:.1f}%)")

# Coverage per asset
print(f"\nCoverage per asset:")
for c in factor.columns:
    cv = factor[c].notna().sum() / len(factor)
    print(f"  {c}: {cv*100:.1f}% ({factor[c].notna().sum()}/{len(factor)})")

print(f"\nFull IC Analysis at 10d:")
fwd_ret = price_df.pct_change(10).shift(-10)
common_idx = factor.dropna(how='all').index.intersection(fwd_ret.dropna(how='all').index)

ics = []
n_assets = []
dates_list = []
for dt in common_idx:
    f = factor.loc[dt]; r = fwd_ret.loc[dt]
    mask = r.notna() & f.notna() & ~np.isinf(r) & ~np.isinf(f)
    if mask.sum() >= 8:
        fv = f[mask].values; rv = r[mask].values
        ic, _ = scipy_stats.spearmanr(fv, rv)
        ics.append(ic)
        n_assets.append(mask.sum())
        dates_list.append(dt)

avg_ic = np.mean(ics)
std_ic = np.std(ics) if len(ics) > 1 else 0.001
icir = avg_ic / std_ic if std_ic > 0 else 0
hit = np.mean([1 for ic in ics if ic > 0])

print(f"  Dates: {len(ics)}")
print(f"  Avg IC: {avg_ic:.4f}")
print(f"  Std IC: {std_ic:.4f}")
print(f"  ICIR: {icir:.4f}")
print(f"  Hit ratio: {hit:.4f}")
print(f"  Avg assets: {np.mean(n_assets):.1f}")
print(f"  Min assets: {min(n_assets)}")

# Check that IC values are not constant
unique_ic_values = len(set(round(ic, 6) for ic in ics))
print(f"  Unique IC values: {unique_ic_values} out of {len(ics)} dates")