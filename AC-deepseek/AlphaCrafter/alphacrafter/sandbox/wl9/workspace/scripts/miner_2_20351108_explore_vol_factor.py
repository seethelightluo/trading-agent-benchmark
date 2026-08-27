"""
Explore volume-based factor: volume_z_score_20d
Volume z-score measures how far current volume deviates from its 20d mean.
Rationale: In cross-asset markets, abnormal volume often signals regime changes,
institutional flows, and sentiment shifts that predict near-term momentum.
"""
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from alphacrafter.sim.utils import (
    get_stock_daily_data,
    get_index_daily_data,
    get_account_dict,
)

# Get account info for watchlist
acct = get_account_dict()
watch_list = acct.get("watch_list", [])
print(f"Watch_list: {watch_list}")
print(f"Number of instruments: {len(watch_list)}")

# Fetch data for all instruments
N_DAYS = 300
all_data = {}
min_dates = {}

for sym in watch_list:
    df = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df is not None and len(df) > 30:
        all_data[sym] = df
        min_dates[sym] = df['date'].min()
        print(f"{sym}: {len(df)} days, from {df['date'].min()} to {df['date'].max()}")
    else:
        print(f"{sym}: No/insufficient data")

if not all_data:
    print("No data available, aborting")
    sys.exit(0)

# Compute volume z-score (20d rolling z-score of volume)
factor_data = {}
for sym, df in all_data.items():
    df = df.copy()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_std20'] = df['volume'].rolling(20).std()
    df['vol_z'] = (df['volume'] - df['volume_ma20']) / df['volume_std20'].clip(lower=1e-8)
    factor_data[sym] = df[['date', 'vol_z', 'close']].dropna()

print(f"\nComputing factor values and IC analysis...")

# Horizon H=10 days (standard for this universe)
H = 10

# Collect panel data
panel_rows = []
for sym, df in factor_data.items():
    for i in range(len(df) - H):
        row = df.iloc[i]
        fwd_row = df.iloc[i + H]
        fwd_return = fwd_row['close'] / row['close'] - 1.0
        panel_rows.append({
            'date': row['date'],
            'symbol': sym,
            'factor': row['vol_z'],
            'fwd_ret': fwd_return
        })

panel = pd.DataFrame(panel_rows)
print(f"Panel: {len(panel)} observations")
print(f"Panel date range: {panel['date'].min()} to {panel['date'].max()}")

# Cross-sectional IC per date
ic_values = []
n_assets_per_date = []

for date, grp in panel.groupby('date'):
    n = len(grp)
    if n >= 8:
        corr = grp['factor'].corr(grp['fwd_ret'])
        if not np.isnan(corr):
            ic_values.append(corr)
            n_assets_per_date.append(n)

ic_series = pd.Series(ic_values)
print(f"\n=== Volume Z-Score Factor H={H} ===")
print(f"Number of IC observations: {len(ic_series)}")
print(f"Mean IC: {ic_series.mean():.6f}")
print(f"Std IC: {ic_series.std():.6f}")
print(f"ICIR: {ic_series.mean() / max(ic_series.std(), 1e-10) * np.sqrt(len(ic_series)):.6f}")
print(f"IC > 0: {(ic_series > 0).sum()}/{len(ic_series)} ({(ic_series > 0).mean()*100:.1f}%)")
print(f"IC Hit Ratio (>|0.01|): {(abs(ic_series) > 0.01).mean()*100:.1f}%")

# Coverage
n_total_dates = panel['date'].nunique()
n_good_dates = len(ic_series)
print(f"\nCoverage: {n_good_dates}/{n_total_dates} dates have >=8 instruments")

# Turnover analysis
print("\n=== Turnover Analysis ===")
turnover_dates = []
for date in sorted(panel['date'].unique()):
    sub = panel[panel['date'] == date][['symbol', 'factor']].dropna()
    if len(sub) >= 8:
        sub = sub.copy()
        sub['rank'] = sub['factor'].rank(pct=True)
        turnover_dates.append((date, sub.set_index('symbol')['rank'].to_dict()))

turnovers = []
for i in range(1, len(turnover_dates)):
    prev = turnover_dates[i-1][1]
    curr = turnover_dates[i][1]
    common = set(prev.keys()) & set(curr.keys())
    if len(common) >= 8:
        prev_r = np.array([prev[s] for s in common])
        curr_r = np.array([curr[s] for s in common])
        turnover = np.mean(np.abs(curr_r - prev_r))
        turnovers.append(turnover)

if turnovers:
    mean_to = np.mean(turnovers)
    print(f"Mean daily rank turnover: {mean_to:.4f}")
    print(f"Estimated 10d turnover (scaled): {mean_to * 10:.4f}")
else:
    print("No turnover data")

# Decay analysis (IC at different horizons)
print("\n=== Decay Analysis ===")
for horizon in [1, 2, 3, 5, 10, 20]:
    h_rows = []
    for sym, df in factor_data.items():
        for i in range(len(df) - horizon):
            row = df.iloc[i]
            fwd_row = df.iloc[i + horizon]
            fwd_return = fwd_row['close'] / row['close'] - 1.0
            h_rows.append({
                'date': row['date'],
                'symbol': sym,
                'factor': row['vol_z'],
                'fwd_ret': fwd_return
            })
    h_panel = pd.DataFrame(h_rows)
    h_ics = []
    for date, grp in h_panel.groupby('date'):
        if len(grp) >= 8:
            corr = grp['factor'].corr(grp['fwd_ret'])
            if not np.isnan(corr):
                h_ics.append(corr)
    if h_ics:
        mean_ic = np.mean(h_ics)
        std_ic = np.std(h_ics)
        h_icir = mean_ic / max(std_ic, 1e-10) * np.sqrt(len(h_ics))
        print(f"H={horizon:2d}: IC={mean_ic:.6f}, ICIR={h_icir:.6f}, n={len(h_ics)}")

print("\n=== Summary ===")
print(f"Volume Z-Score 20d (H=10): IC={ic_series.mean():.6f}, ICIR={ic_series.mean()/max(ic_series.std(),1e-10)*np.sqrt(len(ic_series)):.6f}")
print(f"Admission gates: abs(IC) >= 0.0070 AND abs(ICIR) >= 0.0840")
abs_ic = abs(ic_series.mean())
abs_icir = abs(ic_series.mean() / max(ic_series.std(), 1e-10) * np.sqrt(len(ic_series)))
print(f"abs(IC)={abs_ic:.6f} {'PASS' if abs_ic >= 0.0070 else 'FAIL'}")
print(f"abs(ICIR)={abs_icir:.6f} {'PASS' if abs_icir >= 0.0840 else 'FAIL'}")