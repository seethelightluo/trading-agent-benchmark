"""Trader regime assessment as of 2035-06-01 (data through 2035-05-31)."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
          'COPPER','WTI','BTC','ETH','US10Y','CN10Y']

frames = {}
for a in ASSETS:
    df = get_stock_daily_data(symbol=a, days=170)
    if df is None or len(df) < 30:
        frames[a] = None
        continue
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    frames[a] = df

# 20d / 60d equal-weight cumulative returns
rets20, rets60 = [], []
for a, df in frames.items():
    if df is None or len(df) < 70:
        continue
    c = df['close'].astype(float)
    r20 = c.iloc[-1] / c.iloc[-21] - 1.0
    r60 = c.iloc[-1] / c.iloc[-61] - 1.0
    rets20.append(r20)
    rets60.append(r60)
    print(f"{a:10s} 20d {r20*100:7.2f}%  60d {r60*100:7.2f}%")

print("\nEQW 20d cum: %.2f%% (mean daily %.4f%%)" % (np.mean(rets20)*100, np.mean(rets20)/20*100))
print("EQW 60d cum: %.2f%%" % (np.mean(rets60)*100))

# breadth
n_above20 = sum(1 for a, df in frames.items() if df is not None and len(df) >= 25
                and float(df['close'].iloc[-1]) > float(df['close'].rolling(20).mean().iloc[-1]))
n_above60 = sum(1 for a, df in frames.items() if df is not None and len(df) >= 65
                and float(df['close'].iloc[-1]) > float(df['close'].rolling(60).mean().iloc[-1]))
print("breadth above MA20: %d/15, above MA60: %d/15" % (n_above20, n_above60))

# vol + dispersion
drets = []
for a, df in frames.items():
    if df is None or len(df) < 25:
        continue
    r = df['close'].pct_change().tail(20)
    drets.append(r)
m = pd.concat(drets, axis=1).dropna(how='all')
vol = m.std().mean() * np.sqrt(252)
disp = m.mean(axis=1).std()  # cross-sectional dispersion of daily returns
print("mean 20d ann vol: %.1f%%" % (vol*100))
print("20d x-sect dispersion (mean daily): %.2f%%" % (disp*100))

# VIX
vix = pd.read_csv('../persistent/index_data/VIX.csv')
vix['date'] = pd.to_datetime(vix['date'])
vix = vix[vix['date'] <= pd.Timestamp('2035-05-31')].sort_values('date')
vc = vix['close'].astype(float)
print("\nVIX last:", round(float(vc.iloc[-1]),1), "10d ago:", round(float(vc.iloc[-11]),1),
      "20d ago:", round(float(vc.iloc[-21]),1), "60d ago:", round(float(vc.iloc[-61]),1))

# 20d leaders/laggards
order = sorted(zip(rets20, ASSETS), reverse=True)
print("\n20d leaders:", [(a, round(r*100,1)) for r, a in order[:5]])
print("20d laggards:", [(a, round(r*100,1)) for r, a in order[-5:]])
