"""
Mineral exploration: short-term reversal factor across the 15-asset cross-asset universe.
Idea: after a sharp 5-day move (up or down) in an asset, short-horizon continuation vs reversal?
Existing mom_10d_skip5 skips the most recent 5 days to avoid short-term reversal, implying
reversal effect exists at 1-5 day horizon. Test: fwd 10-day (2-week, matching ensemble
rebalance horizon) return vs trailing 1/2/3/5-day total return.

Construction (interpretable): rev_k = -sign-based: rev = -(close/close.shift(k) - 1) for k in {1,2,3,5}.
We test IC of each raw trailing return and of the negated series separately; also a pure
rank-based version. Admission gate: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 (daily cross-sectional,
15-asset universe, >=8 valid assets per date).
Validation window: 2024-01-01 .. last trading day (2028-03-22) to assess recent robustness,
plus full sample for reference.
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
    """Daily cross-section IC of factor vs fwd return. Returns series of IC."""
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
    s = pd.Series(ics).sort_index()
    return s

def summarize(name, ic_series, full_ic_series):
    if len(ic_series) < 10:
        print(f'{name}: insufficient IC dates ({len(ic_series)})')
        return
    m = ic_series.mean(); sd = ic_series.std()
    icir = m/sd if sd > 0 else 0.0
    hit = (ic_series > 0).mean()
    f_m = full_ic_series.mean(); f_sd = full_ic_series.std()
    f_icir = f_m/f_sd if f_sd > 0 else 0.0
    print(f'{name}: test IC={m:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic_series)} | full IC={f_m:+.4f} ICIR={f_icir:+.4f} n={len(full_ic_series)}')

data = load()
print('loaded assets:', len(data))

# Build aligned close panel
close = pd.DataFrame(data).sort_index()

# Forward 10-day return
fwd = close.shift(-10) / close - 1.0

for k in [1, 2, 3, 5]:
    ret = close / close.shift(k) - 1.0
    # raw trailing return as factor (positive momentum hypothesis)
    ic_raw = rolling_ic(ret, fwd)
    ic_full = rolling_ic(ret.loc[ret.index >= '2020-06-01'], fwd.loc[fwd.index >= '2020-06-01'])
    summarize(f'rev_neg{int(k)}d (=-ret) factor: IC of -ret', -ic_raw, -ic_full)
    summarize(f'ret_{k}d raw momentum view', ic_raw, ic_full)

# Combined: 3-day reversal
ret3 = close / close.shift(3) - 1.0
ret5 = close / close.shift(5) - 1.0
combo = -(ret3 * 0.5 + ret5 * 0.5)
ic_c = rolling_ic(combo, fwd); ic_cf = rolling_ic(combo.loc[combo.index >= '2020-06-01'], fwd.loc[fwd.index >= '2020-06-01'])
summarize('combo rev(3,5) avg', ic_c, ic_cf)