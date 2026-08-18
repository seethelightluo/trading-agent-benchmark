"""Trader probe: current regime + factor scores + naive weights as of 2026-12-16."""
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WL = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
N = 300

def load(a):
    try:
        df = get_stock_daily_data(a, days=N)
        if df is None or len(df) == 0:
            df = get_index_daily_data(a, days=N)
    except Exception:
        df = get_index_daily_data(a, days=N)
    return df

frames = {a: load(a) for a in WL}
closes = {a: f['close'].astype(float) for a, f in frames.items() if f is not None and len(f) >= 140}
usable = [c.rename(a) for a, c in closes.items()]
panel = pd.concat(usable, axis=1, join='inner')
print('panel shape:', panel.shape, 'last date:', panel.index[-1])

# regime
mkt = panel.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 if v20 and v20 > 1e-12 else 0.0
print('20d drift:', round(r20, 5), '20d vol:', round(v20, 5), 'trend t-stat:', round(trend, 3))
regime = 'bull' if trend > 0.25 else ('bear' if trend < -0.25 else 'sideways')
print('regime:', regime)

# individual asset 20d returns
ret20 = panel.pct_change(20).tail(1).iloc[0]
print('\n20d returns:')
for a in WL:
    if a in ret20.index:
        print(f'  {a:10s} {ret20[a]*100:7.2f}%')

# vol of vol and low vol
ret = panel.pct_change()
print('\nvol stats (20d std, vol-of-vol):')
for a in WL:
    if a in ret.columns:
        s = ret[a].rolling(20).std().iloc[-1]
        vov = ret[a].rolling(20).std().rolling(60).std().iloc[-1]
        print(f'  {a:10s} vol20={s*100:6.2f}%  vov={vov*100:6.2f}%')

# VIX beta
vf = get_index_daily_data('VIX', days=N)
vix_ret = vf['close'].astype(float).pct_change() if vf is not None else None
print('\nVIX beta (60d):')
if vix_ret is not None:
    for a in WL:
        c = closes.get(a)
        if c is None: continue
        z = pd.concat([c.pct_change().rename('a'), vix_ret.rename('v')], axis=1).dropna().tail(60)
        if len(z) >= 30 and z['v'].var() > 1e-14:
            b = -float(z['a'].cov(z['v']) / z['v'].var())
            print(f'  {a:10s} neg_vix_beta={b:7.3f}')
