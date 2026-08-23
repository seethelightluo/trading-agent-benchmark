"""
Factor exploration: drawdown gap factor across the 15-asset cross-asset universe.
Idea: after a drawdown from the rolling high, do assets at large discount to their
52-week high tend to recover (mean-reversion over the 10-day rebalance horizon)?
Construction: dd_gap_k = -1 * drawdown_k where drawdown_k = close/rolling_max(close,k) - 1
i.e. positive gap = deeper discount. Test k in {40, 60, 90, 120, 250}.
Contrast with days_since_high_60 (already persisted) and mom_* (which use raw returns).
Admission gate: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 daily cross-sectional, 15 assets, >=8 valid.
Window: 2024-01-01..2028-04-19 test; full sample 2020-06-01.. for reference.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
TEST_START = '2024-01-01'

def load(days=2300):
    out = {}
    for s in WATCHLIST:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is not None and len(df) > 300:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            out[s] = df['close']
    return out

def rolling_ic(factor_df, fwd_df, min_assets=8):
    ics = {}
    common_dates = factor_df.index.intersection(fwd_df.index)
    for dt in common_dates:
        f = factor_df.loc[dt].dropna()
        r = fwd_df.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_assets:
            continue
        x = f[common].values.astype(float)
        y = r[common].values.astype(float)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        ics[dt] = np.corrcoef(x, y)[0, 1]
    return pd.Series(ics).sort_index()

def summarize(name, ic_series, full_ic_series):
    if len(ic_series) < 10:
        print(f'{name}: insufficient IC dates ({len(ic_series)})')
        return None
    m = ic_series.mean(); sd = ic_series.std()
    icir = m/sd if sd > 0 else 0.0
    hit = (ic_series > 0).mean()
    f_m = full_ic_series.mean(); f_sd = full_ic_series.std()
    f_icir = f_m/f_sd if f_sd > 0 else 0.0
    print(f'{name}: test IC={m:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic_series)} | full IC={f_m:+.4f} ICIR={f_icir:+.4f} n={len(full_ic_series)}')
    return m, icir

data = load()
print('loaded assets:', len(data))
close = pd.DataFrame(data).sort_index()

# fwd 10-day return
fwd = close.shift(-10) / close - 1.0

for k in [20, 40, 60, 120, 250]:
    rolling_high = close.rolling(k, min_periods=k).max()
    dd = close / rolling_high - 1.0        # <= 0
    gap = -dd                              # >= 0, deeper drawdown -> higher
    s = summarize(f'dd_gap_{k}d', rolling_ic(gap, fwd), rolling_ic(gap.loc[gap.index>='2020-06-01'], fwd.loc[fwd.index>='2020-06-01']))

# Also variant: normalized by realized vol (gap per unit risk)
vol10 = close.pct_change().rolling(10).std()
for k in [60, 120]:
    dd = close / close.rolling(k, min_periods=k).max() - 1.0
    gap_n = -dd / vol10
    summarize(f'dd_gap_{k}_volnorm', rolling_ic(gap_n, fwd), rolling_ic(gap_n.loc[gap_n.index>='2020-06-01'], fwd.loc[fwd.index>='2020-06-01']))