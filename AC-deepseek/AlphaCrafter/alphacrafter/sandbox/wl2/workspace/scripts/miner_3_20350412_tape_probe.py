"""miner_3 2035-04-12 tape probe: data through 2035-04-11 (visible).
Snapshot of regime to design new factor candidates."""
import sys
sys.path.insert(0, 'scripts')
import pandas as pd
import numpy as np
from miner_3_20260813_lib import load_asset, load_macro, ASSETS, GRID, VISIBLE

print("visible_through:", VISIBLE, "grid rows:", len(GRID))

series = {}
for s in ASSETS:
    df = load_asset(s)
    if df is None or len(df) < 100:
        print("NO DATA:", s)
        continue
    series[s] = df
print("assets with data:", sorted(series.keys()))

panel = pd.DataFrame({s: df["close"] for s, df in series.items()})
panel = panel.reindex(GRID)
ret = panel.pct_change()
last = panel.index[-1]
print("panel dates:", panel.index[0], "->", last, "rows:", len(panel))

def r(days):
    return (panel.iloc[-1] / panel.iloc[-1 - days] - 1.0) * 100

for days in [5, 10, 20, 60, 180, 252]:
    rr = r(days).sort_values(ascending=False)
    print(f"\n=== {days}d return (%) ===")
    print(rr.round(2).to_string())

vix = load_macro('VIX')
print("\nVIX last 5:", vix.tail(5).round(1).to_dict())
print("VIX 20d mean:", round(vix.tail(20).mean(), 1), "60d max:", round(vix.tail(60).max(), 1))

for m in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    s = load_macro(m)
    if s is None:
        continue
    print(f"{m} 20d chg %:", round((s.iloc[-1]/s.iloc[-21]-1)*100, 2))

r20 = r(20)
print("\n20d dispersion (max-min pp):", round(r20.max()-r20.min(), 1))
hp = panel.rolling(252, min_periods=60).max()
lp = panel.rolling(252, min_periods=60).min()
print("\n252d range position (last):")
print(((panel.iloc[-1] - lp.iloc[-1]) / (hp.iloc[-1] - lp.iloc[-1])).round(3).to_string())
