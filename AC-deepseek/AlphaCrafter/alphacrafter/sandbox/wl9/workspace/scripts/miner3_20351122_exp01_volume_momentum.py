"""
Factor: Volume-Confirmed Momentum (vwr_mom_20)
Idea: Momentum signals confirmed by above-average volume are more reliable.
Construction: sign(20d return) * (volume / volume_ma_20) * |20d return|
= ret_20 * (vol / vol_ma_20)
Rationale: price moves with strong volume indicate conviction; low-volume moves are noise.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

acct = get_account_dict()
watch_list = list(acct.get('watch_list', []))
print(f"Watchlist ({len(watch_list)}): {watch_list}")

# Fetch data
data = {}
for sym in watch_list:
    df = get_stock_daily_data(sym, 300)
    if df is not None and len(df) >= 60:
        data[sym] = df
        print(f"  {sym}: {df['date'].min().date()} to {df['date'].max().date()}, {len(df)} rows")
    else:
        data[sym] = None
        print(f"  {sym}: insufficient data")

# Compute factor values: price_ret_20 * (volume / vol_ma_20)
factor_id = "vwr_mom_20"
factor_name = "Volume-Weighted Return Momentum 20d"
records = []

for sym in watch_list:
    df = data[sym]
    if df is None or len(df) < 40:
        continue
    closes = df['close'].values
    volumes = df['volume'].values.astype(float)
    dates = df['date'].values
    
    for i in range(40, len(closes)):
        ret_20 = closes[i] / closes[i-20] - 1.0
        vol_ma = np.mean(volumes[i-20:i])
        if vol_ma > 0:
            vol_ratio = volumes[i] / vol_ma
        else:
            vol_ratio = 1.0
        factor_val = ret_20 * vol_ratio  # volume-confirmed momentum
        dt = pd.Timestamp(dates[i])
        records.append((dt, sym, factor_val, closes[i], ret_20, vol_ratio))

df_f = pd.DataFrame(records, columns=['date', 'asset', 'factor', 'close', 'ret_20', 'vol_ratio'])
df_f['date'] = pd.to_datetime(df_f['date'])
print(f"\nFactor records: {len(df_f)}")
print(f"Date range: {df_f['date'].min()} to {df_f['date'].max()}")
print(f"Factor stats:\n{df_f['factor'].describe()}")

# Forward returns (10d)
df_s = df_f.sort_values(['asset', 'date'])
df_s['fwd_ret_10'] = df_s.groupby('asset')['close'].transform(lambda x: x.shift(-10) / x - 1.0)
valid = df_s.dropna(subset=['fwd_ret_10'])
print(f"Valid rows with fwd_ret: {len(valid)}")

# Cross-sectional IC per date
ic_list = []
n_ge8 = 0
for dt, grp in valid.groupby('date'):
    if len(grp) < 8:
        continue
    n_ge8 += 1
    f = grp['factor'].values
    r = grp['fwd_ret_10'].values
    if np.std(f) > 1e-10 and np.std(r) > 1e-10:
        rho = np.corrcoef(f, r)[0, 1]
        ic_list.append(rho)

ic_arr = np.array(ic_list)
print(f"\n=== Validation Results ===")
print(f"Valid IC dates (>=8 assets): {n_ge8}")
print(f"Number of IC observations: {len(ic_arr)}")
if len(ic_arr) > 0:
    mean_ic = np.mean(ic_arr)
    std_ic = np.std(ic_arr)
    icir = mean_ic / std_ic if std_ic > 0 else 0
    hit_ratio = np.mean(ic_arr > 0)
    print(f"IC (mean): {mean_ic:.6f}")
    print(f"IC (std):  {std_ic:.6f}")
    print(f"ICIR:      {icir:.6f}")
    print(f"Hit ratio: {hit_ratio:.4f}")
    
    # Coverage
    cov_asset_days = len(valid) / (len(watch_list) *