"""miner_2 debug: per-asset coverage and panel construction shapes (2026-11-19 cycle)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2026-11-04')
ASSETS = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225',
          'NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']

def load_close(sym, base):
    df = pd.read_csv(base / f'{sym}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
    return df['close'].astype(float)

px = pd.DataFrame({a: load_close(a, Path('../persistent/stock_data')) for a in ASSETS}).sort_index()
vix = load_close('VIX', Path('../persistent/index_data'))
dxy = load_close('DXY', Path('../persistent/index_data'))
ret = px.pct_change()

print('px shape', px.shape, 'first/last', px.index[0].date(), px.index[-1].date())
print('\nper-asset non-NaN counts (px):')
for a in ASSETS:
    s = px[a].dropna()
    print(f'  {a:10s} n={len(s):5d} {s.index[0].date()}..{s.index[-1].date()}')
print('\nper-asset coverage of ret after 2021-01-01:')
r21 = ret.loc['2021-01-01':]
for a in ASSETS:
    print(f'  {a:10s} {float(r21[a].notna().mean()):.3f}')

# replicate factor panels and print shapes
s5, s15, s125 = px.shift(5), px.shift(15), px.shift(125)
panels = {}
panels['mom_10d_skip5'] = s5 / s15 - 1.0
panels['mom_120d_skip5'] = s5 / s125 - 1.0
panels['vol_of_vol20x60'] = ret.rolling(20).std(ddof=0).rolling(60).std(ddof=0)

def rolling_beta(x_df, y_s, w):
    my = y_s.rolling(w).mean()
    cov = (x_df * y_s).rolling(w).mean() - x_df.rolling(w).mean() * my
    var = (y_s ** 2).rolling(w).mean() - my ** 2
    return cov / var

v20 = rolling_beta(ret, vix.pct_change(), 60)
print('\nv20 shape', v20.shape, 'columns[:5]', list(v20.columns)[:5], 'nonNaN', float(v20.notna().sum().sum()))
vix_move = vix / vix.shift(20) - 1.0
vbc = -v20.mul(vix_move, axis=0)
print('vbc shape', vbc.shape, 'columns[:5]', list(vbc.columns)[:5], 'nonNaN', float(vbc.notna().sum().sum()))

rsi = 100.0 - 100.0 / (1.0 + ret.clip(lower=0).rolling(14).mean() / (-ret.clip(upper=0).rolling(14).mean()))
print('\nrsi shape', rsi.shape, 'nonNaN', float(rsi.notna().sum().sum()))

for fid, p in panels.items():
    print(fid, 'shape', p.shape, 'columns[:3]', list(p.columns)[:3], 'nonNaN', float(p.notna().sum().sum()))
