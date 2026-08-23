"""
Factor exploration: Close Location Value (CLV) — intraday buying/selling pressure.
Idea: CLV_t = (close - low) / (high - low). A close near the high (CLV -> 1) means
buyers dominated the session; a close near the low (CLV -> 0) means sellers dominated.
Trailing-window average CLV measures persistent directional pressure over recent days.
Hypothesis (test both directions): assets with sustained high CLV (bullish pressure)
either persist (momentum of pressure, positive fwd returns) or mean-revert (negative).

Construction: clv_k = rolling_mean((close-low)/(high-low), k), k in {5,10,20,40}.
Admission gate: abs(IC) >= 0.0070 and abs(ICIR) >= 0.0840 daily cross-sectional,
15 tradable assets, >=8 valid instruments per date.
Test window: 2024-01-01..2028-07-26 (recent). Full sample 2020-01-01..2028-07-26 reference.
Metrics: Spearman rank IC vs forward returns; ICIR; hit ratio; coverage; turnover; decay.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
TEST_START = '2024-01-01'
MIN_ASSETS = 8

def load(days=2400):
    out = {}
    for s in WATCHLIST:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is not None and len(df) > 300:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            out[s] = df
    return out

def rolling_ic(factor_df, fwd_df, min_assets=MIN_ASSETS):
    ics = {}
    common = factor_df.index.intersection(fwd_df.index)
    for dt in common:
        f = factor_df.loc[dt]
        r = fwd_df.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() < min_assets:
            continue
        x = f[m].values.astype(float); y = r[m].values.astype(float)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        rho, _ = spearmanr(x, y)
        if np.isfinite(rho):
            ics[dt] = rho
    return pd.Series(ics).sort_index()

def forward_returns(close, h):
    return close.shift(-h) / close - 1.0

def summarize(label, ic_test, ic_full, sign=1.0):
    if len(ic_test) < 20:
        print(f'{label}: insufficient test IC dates ({len(ic_test)})'); return None
    m = sign*ic_test.mean(); sd = ic_test.std(ddof=1)
    icir = m/sd if sd > 0 else np.nan
    hit = (sign*ic_test > 0).mean()
    f_m = sign*ic_full.mean(); f_sd = ic_full.std(ddof=1)
    f_icir = f_m/f_sd if f_sd > 0 else np.nan
    f_hit = (sign*ic_full > 0).mean()
    print(f'{label}: test IC={m:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic_test)} | '
          f'full IC={f_m:+.4f} ICIR={f_icir:+.4f} hit={f_hit:.3f} n={len(ic_full)}')
    return m, icir

data = load()
print('loaded assets:', len(data))
close = pd.DataFrame({s: df['close'] for s, df in data.items()}).sort_index()

# CLV per asset
clv = {}
for s, df in data.items():
    hl = (df['high'] - df['low'])
    clv[s] = ((df['close'] - df['low']) / hl.replace(0, np.nan)).clip(0, 1)
clv_df = pd.DataFrame(clv).sort_index()
print('clv_df shape:', clv_df.shape)

fwd10 = forward_returns(close, 10)
print('fwd10 range:', fwd10.dropna(how='all').index.min().date(), '->',
      fwd10.dropna(how='all').index.max().date())

print('\n=== CLV averaged over trailing windows, fwd horizon 10 (admission) ===')
best = None
for k in [5, 10, 20, 40]:
    fac = clv_df.rolling(k).mean()
    ic10 = rolling_ic(fac, fwd10)
    ic10_test = ic10[ic10.index >= TEST_START]
    res = summarize(f'clv_avg{k}_fwd10', ic10_test, ic10)
    if res:
        icir_full = res[1]
        # also negative-sign variant (mean reversion) reported as abs pass/fail
        if res[0] < 0: print(f'   -> negative direction candidate (abs IC={abs(res[0]):.4f} abs ICIR={abs(icir_full):.4f})')

print('\n=== decay profile for k=10 (both sign conventions) ===')
fac = clv_df.rolling(10).mean()
for h in [1, 2, 3, 5, 10, 20]:
    fh = forward_returns(close, h)
    ic_h_full = rolling_ic(fac, fh)
    ic_h_test = ic_h_full[ic_h_full.index >= TEST_START]
    if len(ic_h_test):
        rho, _ = spearmanr(np.arange(len(ic_h_test)), ic_h_test.values)
        trend = f'trend_p={rho:.3f}'
    else:
        trend = ''
    print(f'  h={h:2d}: full IC={ic_h_full.mean():+.4f} (n={len(ic_h_full)}) | test IC={ic_h_test.mean():+.4f} (n={len(ic_h_test)}) {trend}')

print('\n=== turnover & coverage (k=10) ===')
fac = clv_df.rolling(10).mean()
ranks = fac.rank(axis=1)
chg = ranks.diff(10).abs().mean(axis=1)
n_valid_dates = (fac.notna().sum(axis=1) >= MIN_ASSETS)
print(f'coverage asset-days: {fac.notna().sum().sum()/(fac.shape[0]*fac.shape[1]):.4f} | '
      f'dates>=8 assets: {n_valid_dates.mean():.3f} ({n_valid_dates.sum()} dates) | '
      f'avg valid assets/date: {fac.notna().sum(axis=1).mean():.1f}')
print(f'mean 10d rank change: {chg.mean():.3f} | median: {chg.median():.3f}')