"""
Miner2: Explore "Cross-Asset Range Contraction/Expansion" factor
Date: 2035-11-22
Idea: The ratio of recent high-low range to medium-term average range signals
volatility regime changes. Contracting ranges often precede breakouts, while
expanding ranges coincide with trend establishment or panic.
Construction: range_ratio = (high - low).rolling(5).mean() / (high - low).rolling(20).mean()
"""
import json
import sys
import math
import pandas as pd
import numpy as np
from alphacrafter.sim.utils import (
    get_stock_daily_data,
    get_index_daily_data,
    get_account_dict,
)
from scipy.stats import spearmanr

# --- Config ---
CURRENT_DATE_STR = "2035-11-22"
MIN_WINDOW = 60  # need at least 60 days for 20d rolling
FWD_HORIZON = 10  # 10-day forward return
MIN_VALID_DATES = 30
MIN_VALID_ASSETS = 8
IC_THRESHOLD = 0.0070
ICIR_THRESHOLD = 0.0840

# --- Get account & watchlist ---
acct = get_account_dict()
watch_list = list(acct.get("watch_list", []))
print(f"Watch list ({len(watch_list)} assets): {watch_list}")

# --- Fetch data ---
data = {}
for sym in watch_list:
    df = get_stock_daily_data(symbol=sym, days=750)  # ~3 years
    if df is not None and len(df) >= MIN_WINDOW:
        df = df.copy()
        df['symbol'] = sym
        data[sym] = df
        print(f"{sym}: {len(df)} rows, {df.index[0]}..{df.index[-1]}")
    else:
        print(f"{sym}: insufficient data or None ({len(df) if df is not None else 0})")

if len(data) < MIN_VALID_ASSETS:
    print(f"ERROR: Only {len(data)} assets with data, need {MIN_VALID_ASSETS}")
    sys.exit(1)

# --- Build panel ---
# Align all on a common date index
all_dates = None
for sym, df in data.items():
    dates = pd.to_datetime(df['date']).drop_duplicates()
    if all_dates is None:
        all_dates = set(dates)
    else:
        all_dates = all_dates.intersection(set(dates))

all_dates = sorted(all_dates)
print(f"Common dates: {len(all_dates)} ({all_dates[0]}..{all_dates[-1]})")

if len(all_dates) < MIN_WINDOW:
    print(f"ERROR: too few common dates ({len(all_dates)})")
    sys.exit(1)

# --- Compute factor: range_ratio_5_20 ---
factor_df = pd.DataFrame(index=all_dates)
for sym, df in data.items():
    df = df.set_index(pd.to_datetime(df['date']))
    df = df[~df.index.duplicated(keep='last')]
    df = df.reindex(all_dates, method=None)
    
    high_low_range = (df['high'] - df['low']).astype(float)
    range_short = high_low_range.rolling(5, min_periods=3).mean()
    range_long = high_low_range.rolling(20, min_periods=10).mean()
    
    factor = range_short / range_long
    factor = factor.clip(lower=0.1, upper=10.0)  # clip outliers
    factor_df[sym] = factor

# --- Compute forward returns ---
ret_df = pd.DataFrame(index=all_dates)
for sym, df in data.items():
    close = df.set_index(pd.to_datetime(df['date']))['close'].astype(float)
    close = close[~close.index.duplicated(keep='last')]
    close = close.reindex(all_dates, method=None)
    fwd_ret = close.shift(-FWD_HORIZON) / close - 1.0
    ret_df[sym] = fwd_ret

print(f"Factor shape: {factor_df.shape}")
print(f"Factor stats:\n{factor_df.describe()}")

# --- IC analysis ---
ic_values = []
ic_dates_used = []
n_assets_list = []

for date_idx in range(MIN_WINDOW, len(all_dates)):
    date = all_dates[date_idx]
    f_vals = factor_df.loc[date].values
    r_vals = ret_df.loc[date].values
    
    valid = ~(np.isnan(f_vals) | np.isnan(r_vals) | np.isinf(f_vals) | np.isinf(r_vals))
    n_valid = valid.sum()
    
    if n_valid >= MIN_VALID_ASSETS:
        f_clean = f_vals[valid]
        r_clean = r_vals[valid]
        try:
            ic, pval = spearmanr(f_clean, r_clean)
            if not (np.isnan(ic) or np.isinf(ic)):
                ic_values.append(ic)
                ic_dates_used.append(date)
                n_assets_list.append(n_valid)
        except:
            pass

ic_values = np.array(ic_values)
print(f"\n===