"""miner_1 2034-12-07 probe: current tape snapshot (data through 2034-12-06 visible).
Reports recent returns, VIX regime, dispersion, macro to inform factor design.
"""
import sys
sys.path.insert(0, 'scripts')
import pandas as pd
import numpy as np
from miner_1_lib import load_panel, macro_series, TRADABLES

panel = load_panel()
ret = panel.pct_change()
last = panel.index[-1]
print("panel dates:", panel.index[0].date(), "->", last.date(), "rows:", len(panel))

def r(days):
    return (panel.iloc[-1] / panel.iloc[-1 - days] - 1.0) * 100

for days in [5, 10, 20, 60, 180, 252]:
    rr = r(days).sort_values(ascending=False)
    print(f"\n=== {days}d return (%) ===")
    print(rr.round(2).to_string())

vix = macro_series('VIX')
print("\nVIX last 5:", vix.tail(5).round(1).to_dict())
print("VIX 20d mean:", round(vix.tail(20).mean(), 1), "60d max:", round(vix.tail(60).max(), 1))
print("VIX 60d ago:", round(vix.iloc[-61], 1) if len(vix) > 61 else 'NA')

for m in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    s = macro_series(m)
    print(f"{m} 20d chg %:", round((s.iloc[-1]/s.iloc[-21]-1)*100, 2))

r20 = r(20)
print("\n20d dispersion (max-min pp):", round(r20.max()-r20.min(), 1))
hp = panel.rolling(252, min_periods=60).max()
lp = panel.rolling(252, min_periods=60).min()
print("\n252d range position:")
print(((panel.iloc[-1] - lp.iloc[-1]) / (hp.iloc[-1] - lp.iloc[-1])).round(3).to_string())

# last 10-trading-day block returns (approx cycle135 11-23 -> 12-07)
if len(panel) > 11:
    b0 = panel.index[-1]
    b1 = panel.index[-11]
    print("\nlast-10d block return % (approx cycle135 in-block):")
    print(((panel.loc[b0]/panel.loc[b1]-1)*100).round(2).to_string())

# flat-feed check: recent 20d returns of the six known flat names
flat = [a for a in TRADABLES if a in ['000300.SH', 'HSI', 'SX5E', 'BTC', 'US10Y', 'CN10Y']]
print("\nflat-feed candidates 20d abs ret:")
print((panel.iloc[-1]/panel.iloc[-21]-1)[flat].abs().round(4).to_string())
