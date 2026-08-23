"""
Factor exploration: candle-body ratio (decisiveness) across the 15-asset cross-asset universe.
Idea: average daily |close-open|/(high-low) over a trailing window measures how "decisive"
each day's price action is (large body = conviction; small body = indecision/doji).
Hypothesis: assets with recent high decisiveness persist (positive forward 10d returns),
or alternatively reflect over-heated directional moves (negative). We test both directions.
Construction: body_k = rolling_mean(2*|close-open|/(high-low) - 1, k) over k in {10,20,40}.
Admission gate: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 daily cross-sectional, 15 assets, >=8 valid.
Test window: 2024-01-01..2028-07-12 (recent, out-of-ensemble-time). Full sample reference.
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
            out[s] = df
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

def forward_returns(close, h):
    return close.shift(-h) / close - 1.0

def summarize(name, ic_series, full_ic_series, sign=1.0):
    if len(ic_series) < 10:
        print(f'{name}: insufficient IC dates ({len(ic_series)})')
        return None
    m = sign * ic_series.mean(); sd = ic_series.std()
    icir = m/sd if sd > 0 else 0.0
    hit = (sign * ic_series > 0).mean()
    f_m = sign * full_ic_series.mean(); f_sd = full_ic_series.std()
    f_icir = f_m/f_sd if f_sd > 0 else 0.0
    f_hit = (sign * full_ic_series > 0).mean()
    print(f'{name}: test IC={m:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic_series)} | '
          f'full IC={f_m:+.4f} ICIR={f_icir:+.4f} hit={f_hit:.3f} n={len(full_ic_series)}')
    return m, icir

data = load()
print('loaded assets:', len(data))

# Build per-asset candle body ratio series
body = {}
for s, df in data.items():
    hl = (df['high'] - df['low']).replace(0, np.nan)
    raw = (2.0 * (df['close'] - df['open']).abs() / hl - 1.0)
    body[s] = raw
body_df = pd.DataFrame(body).sort_index()

close = pd.DataFrame({s: df['close'] for s, df in data.items()}).sort_index()
fwd10 = forward_returns(close, 10)
fwd5 = forward_returns(close, 5)
print('body_df shape:', body_df.shape, 'close shape:', close.shape)
print('fwd10 sample:', fwd10.dropna(how='all').index[0].date(), '->', fwd10.dropna(how='all').index[-1].date())

for k in [5, 10, 20, 40]:
    fac = body_df.rolling(k).mean()
    ic10 = rolling_ic(fac, fwd10)
    ic10_full = rolling_ic(fac, fwd10)
    # test window subset
    ic10_test = ic10[ic10.index >= TEST_START]
    print(f'--- body ratio k={k} (higher = more decisive) ---')
    summarize(f'body_{k}_fwd10', ic10_test, ic10_full)
    # also test negative direction via sign=-1 (informational only; gate uses abs)
    ic5 = rolling_ic(fac, fwd5)
    ic5_test = ic5[ic5.index >= TEST_START]
    summarize(f'body_{k}_fwd5', ic5_test, ic5)
    # decay check for k=20,40 at h=1,3 for the more promising horizon later

print('\n=== turnover & coverage for best-ish candidates ===')
for k in [10, 20]:
    fac = body_df.rolling(k).mean()
    # rank turnover over 10-day rebalance
    ranks = fac.rank(axis=1)
    chg = ranks.diff(10).abs().mean(axis=1)
    print(f'body_{k}: mean 10d rank change={chg.mean():.3f} (0=stable, higher=turnover) | '
          f'coverage dates>=8: {(fac.notna().sum(axis=1)>=8).mean():.3f} | '
          f'avg valid assets/date: {fac.notna().sum(axis=1).mean():.1f}')