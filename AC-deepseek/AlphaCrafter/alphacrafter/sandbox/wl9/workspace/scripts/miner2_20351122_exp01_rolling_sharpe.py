"""
Miner2: Explore "Rolling Sharpe Ratio" factor
Date: 2035-11-22
Idea: Risk-adjusted momentum using rolling Sharpe ratio (mean/std of returns),
which captures trending with proper volatility normalization.
Construction: sharpe_20d = mean(returns,20) / std(returns,20)
"""
import json, sys, pandas as pd, numpy as np
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
from scipy.stats import spearmanr

CURRENT_DATE_STR = '2035-11-22'
MIN_WINDOW = 60
FWD_HORIZON = 10
MIN_VALID_ASSETS = 8
IC_THRESHOLD = 0.0070
ICIR_THRESHOLD = 0.0840

acct = get_account_dict()
watch_list = list(acct.get('watch_list', []))
print(f'Watch list ({len(watch_list)} assets): {watch_list}')

data = {}
for sym in watch_list:
    df = get_stock_daily_data(symbol=sym, days=1000)
    if df is not None and len(df) >= MIN_WINDOW:
        data[sym] = df.copy()
        print(f'{sym}: {len(df)} rows')

if len(data) < MIN_VALID_ASSETS:
    print(f'ERROR: Only {len(data)} assets')
    sys.exit(1)

all_dates = None
for sym, df in data.items():
    dates = set(pd.to_datetime(df['date']).drop_duplicates())
    if all_dates is None: all_dates = dates
    else: all_dates = all_dates.intersection(dates)
all_dates = sorted(all_dates)
print(f'Common dates: {len(all_dates)} ({all_dates[0]}..{all_dates[-1]})')

# === Rolling Sharpe (20d) ===
print('\n' + '='*70)
print('FACTOR 1: rolling_sharpe_20d')
print('Risk-adjusted momentum: mean(ret,20) / std(ret,20)')
print('='*70)

factor_vals = {}
ret_vals = {}
for sym, df in data.items():
    df = df.set_index(pd.to_datetime(df['date']))
    df = df[~df.index.duplicated(keep='last')]
    df = df.reindex(all_dates)
    
    ret = df['close'].astype(float).pct_change()
    mean_ret = ret.rolling(20, min_periods=10).mean()
    std_ret = ret.rolling(20, min_periods=10).std()
    sharpe = mean_ret / std_ret.replace(0, np.nan)
    sharpe = sharpe.clip(-5, 5)
    factor_vals[sym] = sharpe
    
    fwd_ret = df['close'].astype(float).shift(-FWD_HORIZON) / df['close'].astype(float) - 1.0
    ret_vals[sym] = fwd_ret

f_df = pd.DataFrame(factor_vals)
r_df = pd.DataFrame(ret_vals)

print(f'Factor stats:')
print(f_df.describe().to_string())

ic_values = []
for idx in range(MIN_WINDOW, len(all_dates)):
    date = all_dates[idx]
    fv = f_df.loc[date].values
    rv = r_df.loc[date].values
    valid = ~(np.isnan(fv) | np.isnan(rv) | np.isinf(fv) | np.isinf(rv))
    nv = valid.sum()
    if nv >= MIN_VALID_ASSETS:
        ic, _ = spearmanr(fv[valid], rv[valid])
        if not np.isnan(ic): ic_values.append(ic)

ic_arr = np.array(ic_values)
mean_ic = np.mean(ic_arr)
std_ic = np.std(ic_arr, ddof=1)
icir = mean_ic / std_ic if std_ic > 0 else 0.0
abs_mean_ic = np.mean(np.abs(ic_arr))
print(f'\nResults for rolling_sharpe_20d:')
print(f'  Dates analyzed: {len(ic_arr)}')
print(f'  Mean IC: {mean_ic:.6f}')
print(f'  Std IC: {std_ic:.6f}')
print(f'  ICIR: {icir:.6f}')
print(f'  Abs IC mean: {abs_mean_ic:.6f}')
print(f'  IC > 0: {np.mean(ic_arr>0)*100:.1f}%')
print(f'  Gates: IC_abs={abs_mean_ic >= IC_THRESHOLD}, ICIR_abs={abs(icir) >= ICIR_THRESHOLD}')

# === FACTOR 2: Cross-asset dispersion ===
print('\n' + '='*70)
print('FACTOR 2: cross_asset_dispersion')
print('Low cross-sectional dispersion favors trending regime, high favors mean-reversion')
print('='*70)

ret_20d = pd.DataFrame({})
for sym, df in data.items():
    df = df.set_index(pd.to_datetime(df['date']))
    df = df[~df.index.duplicated(keep='last')]
    df = df.reindex(all_dates)
    ret_20d[sym] = df['close'].astype(float).pct_change(20)

cs_disp = ret_20d.std(axis=1)
# Negative: low dispersion = trending = good
factor_vals2 = {sym: -1.0 * cs_disp for sym in data}
f_df2 = pd.DataFrame(factor_vals2)

ic_values2 = []
for idx in range(MIN_WINDOW, len(all_dates)):
    date = all_dates[idx]
    fv = f_df2.loc[date].values
    rv = r_df.loc[date].values
    valid = ~(np.isnan(fv) | np.isnan(rv) | np.isinf(fv) | np.isinf(rv))
    nv = valid.sum()
    if nv >= MIN_VALID_ASSETS:
        ic, _ = spearmanr(fv[valid], rv[valid])
        if not np.isnan(ic): ic_values2.append(ic)

ic_arr2 = np.array(ic_values2)
mean_ic2 = np.mean(ic_arr2)
std_ic2 = np.std(ic_arr2, ddof=1)
icir2 = mean_ic2 / std_ic2 if std_ic2 > 0 else 0.0
abs_mean_ic2 = np.mean(np.abs(ic_arr2))
print(f'\nResults for cross_asset_dispersion:')
print(f'  Dates analyzed: {len(ic_arr2)}')
print(f'  Mean IC: {mean_ic2:.6f}')
print(f'  Std IC: {std_ic2:.6f}')
print(f'  ICIR: {icir2:.6f}')
print(f'  Abs IC mean: {abs_mean_ic2:.6f}')
print(f'  IC > 0: {np.mean(ic_arr2>0)*100:.1f}%')
print(f'  Gates: IC_abs={abs_mean_ic2 >= IC_THRESHOLD}, ICIR_abs={abs(icir2) >= ICIR_THRESHOLD}')