import json, math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

# Regime at decision 2031-06-06 (data through 0605)
rets20 = {}
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=40)
    if df is None or len(df) < 25: continue
    df = df.sort_values('date')
    rets20[a] = float(df['close'].pct_change().tail(20).mean())
m = float(np.mean(list(rets20.values())))
regime = "bull" if m > 0.010 else ("bear" if m < -0.010 else "side")
print("20d mean daily return @decision:", round(m,5), "regime:", regime)
print("breadth above 0:", sum(1 for v in rets20.values() if v>0), "/15")
print("per-asset 20d mean:", {k: round(v,5) for k,v in sorted(rets20.items(), key=lambda x:x[1])})

# Block PnL per asset (0606 -> 0620), using account book values
with open('../persistent/account.json') as f:
    acc = json.load(f)
tot = acc['net_assets']
print("\nblock-end weights:", {p['symbol']: round(p['market_value']/tot*100,2) for p in acc['positions']})

# Compute per-asset block return using pre-block prices (from memory: 0606 close)
pre_px = {
 '000300.SH': 2843.4112, 'SPX': 5447.3157, 'HSI': 15501.9547, 'N225': 41328.2119,
 'SX5E': 5038.4505, '000688.SH': 796.0935, 'SOX': 4728.4626, 'NDX': 20267.5615,
 'XAU': 5600.4346, 'COPPER': 4.9819, 'WTI': 121.1855, 'BTC': 49118.6271,
 'ETH': 1218.922, 'US10Y': 3.8095, 'CN10Y': 2.0437}
print("\nper-asset block return & contribution:")
contrib = {}
for p in acc['positions']:
    sym = p['symbol']; px0 = pre_px.get(sym)
    if px0:
        r = p['current_price']/px0 - 1
        contrib[sym] = r
        print(f"  {sym:10s} {r*100:+7.2f}%  w_end {p['market_value']/tot*100:5.2f}%")
totc = sum(contrib.values())
print("avg unweighted:", round(totc/len(contrib)*100,2), "%")
