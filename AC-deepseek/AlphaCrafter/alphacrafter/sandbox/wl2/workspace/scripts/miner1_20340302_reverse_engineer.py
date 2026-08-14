"""Reverse-engineer IC/ICIR methodology from admitted factor calmness_20.

Goal: match persisted metrics ic=0.0292, icir=0.0997, n_ic_dates=1665
from the signal artifact calmness_20.signal.npy (dates x assets, calendar-daily).
"""
import numpy as np
import pandas as pd

UNIVERSE = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
            'COPPER','WTI','BTC','ETH','US10Y','CN10Y']

# load closes
closes = {}
for s in UNIVERSE:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')['close']
    closes[s] = df

# Build a calendar-daily index from the union
all_idx = sorted(set().union(*[closes[s].index for s in UNIVERSE]))
close_df = pd.DataFrame(index=all_idx, columns=UNIVERSE, dtype=float)
for s in UNIVERSE:
    close_df[s] = closes[s]

sig = np.load('factors/calmness_20.signal.npy')
print('sig shape', sig.shape)
# provenance: dates_first 2020-01-01, dates_last 2026-07-29 -> align on calendar dates
cal = pd.date_range('2020-01-01', periods=sig.shape[0], freq='D')
print('cal first/last', cal[0], cal[-1], 'n', len(cal))
sig_df = pd.DataFrame(sig, index=cal, columns=UNIVERSE)

# restrict to through 2026-07-29 (the artifact's own last date)
sub = close_df.loc[:'2026-07-29']
print('close rows in window', len(sub))

# Try: forward 10d return using NEXT AVAILABLE observation at least 10 calendar days later?
# First try plain: forward return = close.shift(-10)/close - 1 on calendar grid (with NaN weekends)
def ic_series(factor_df, ret_df, method='spearman', min_valid=8):
    ics = []
    dates = []
    for t in factor_df.index:
        f = factor_df.loc[t]
        r = ret_df.loc[t]
        mask = f.notna() & r.notna()
        if mask.sum() >= min_valid:
            ic = f[mask].corr(r[mask], method=method)
            if np.isfinite(ic):
                ics.append(ic)
                dates.append(t)
    return pd.Series(ics, index=dates)

# candidate forward return definitions
for label, fwd in [
    ('cal_10d', lambda c: c.shift(-10) / c - 1),
    ('cal_5d', lambda c: c.shift(-5) / c - 1),
    ('cal_15d', lambda c: c.shift(-15) / c - 1),
]:
    ret = close_df.apply(fwd, axis=0)
    ics = ic_series(sig_df, ret)
    print(f'forward {label}: n={len(ics)} ic={ics.mean():.4f} icir={ics.mean()/ics.std():.4f} hit={((ics>0).mean()):.3f}')

# candidate: forward return over next trading-day close, h=10 trading days per asset (using per-asset trading days)
# factor aligned on calendar; forward return = close 10 trading days later (per asset) / close today - 1
ret2 = pd.DataFrame(index=close_df.index, columns=UNIVERSE, dtype=float)
for s in UNIVERSE:
    c = close_df[s]
    valid = c.dropna()
    # map each date to return over next 10 valid observations of this asset
    r = c / c.shift(-10) - 1  # calendar shift won't work; do per-asset valid-based
    vals = valid.shift(-10, freq=None)  # not right
    # simpler: build with valid series
    r_series = valid / valid.shift(-10) - 1  # shift within valid-only series, but indexed by its own dates
    ret2.loc[r_series.index, s] = r_series
ics = ic_series(sig_df, ret2)
print(f'forward per-asset 10 trading days: n={len(ics)} ic={ics.mean():.4f} icir={ics.mean()/ics.std():.4f} hit={((ics>0).mean()):.3f}')
