"""Trader cycle analysis for block 2030-09-13 -> 2030-09-27.

Reconstructs the 0913 executed target from current quantities * price@0913 and
computes per-asset block returns/contributions using data visible at 0912/0927.
"""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

acc = json.load(open('../persistent/account.json'))
qty = {p['symbol']: p['quantity'] for p in acc['positions']}
nav_end = acc['net_assets']
nav_start = 884155.52  # observed at 0913 before step

print("=== prices at decision 0913 and block end 0927 ===")
px0913, px0927 = {}, {}
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=170)
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    # 0913 decision sees data through 0912 (previous completed day)
    sub = df[df.index <= '2030-09-12']
    if len(sub):
        px0913[a] = float(sub['close'].iloc[-1])
    sub2 = df[df.index <= '2030-09-27']
    if len(sub2):
        px0927[a] = float(sub2['close'].iloc[-1])
    print(a, 'px@0912', round(px0913.get(a, float('nan')), 4),
          'px@0927', round(px0927.get(a, float('nan')), 4))

print("\n=== executed target @0913 (qty * px@0912) ===")
w = {a: qty.get(a, 0) * px0913.get(a, 0) for a in WATCH}
tot = sum(w.values())
print('implied NAV@0913 from qty*px:', round(tot, 1), 'vs observed', nav_start)
wsum = sum(w.values())
for a in sorted(w, key=lambda x: -w[x]):
    print(f'  {a:10s} {100*w[a]/wsum:6.2f}%  qty={qty.get(a,0):.4f}')

print("\n=== block returns 0913(px@0912)->0927(px@0927) & contributions ===")
contrib = {}
for a in WATCH:
    p0, p1 = px0913.get(a), px0927.get(a)
    if p0 and p1 and p0 > 0:
        r = p1 / p0 - 1.0
        contrib[a] = 100 * r * w[a] / wsum
        print(f'  {a:10s} ret={100*r:7.2f}%  w={100*w[a]/wsum:5.2f}%  contrib={contrib[a]:+6.2f}%')
print('sum contrib:', round(sum(contrib.values()), 2), 'vs block PnL -2.04%')

print("\n=== regime @0912: 20d cross-asset mean daily ret ===")
rets = []
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=170)
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    sub = df[df.index <= '2030-09-12'].tail(25)
    if len(sub) >= 25:
        rets.append(float(sub['close'].pct_change().tail(20).mean()))
m = float(np.mean(rets))
print('20d mean daily ret:', round(m, 5), '-> regime:', 'bull' if m > 0.010 else ('bear' if m < -0.010 else 'side'))

# MA20 / momentum state at decision
print("\n=== trend state @0912 (below MA20 / mom120d) ===")
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=170)
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    sub = df[df.index <= '2030-09-12']
    if len(sub) < 130:
        print(a, 'insufficient'); continue
    c = sub['close'].astype(float)
    ma20 = float(c.rolling(20).mean().iloc[-1])
    last = float(c.iloc[-1])
    mom = float(c.shift(5).iloc[-1] / c.shift(125).iloc[-1] - 1.0)
    print(f'  {a:10s} belowMA20={last<ma20}  mom120d={100*mom:7.1f}%')
