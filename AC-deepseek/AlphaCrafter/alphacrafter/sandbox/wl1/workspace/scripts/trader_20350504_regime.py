"""Regime probe as of 2035-05-04 (data through 2035-05-03) for trader summary."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

acc = get_account_dict()
assets = acc.get("watch_list", [])
frames = {}
for a in assets:
    df = get_stock_daily_data(symbol=a, days=130)
    if df is None or len(df) < 30:
        frames[a] = None
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    frames[a] = df.set_index("date").sort_index()

# 20d / 60d / 120d returns, MA20 breadth
rets20, rets60, rets120 = {}, {}, {}
above_ma20 = 0
for a, df in frames.items():
    if df is None or len(df) < 125:
        continue
    c = df["close"].astype(float)
    rets20[a] = float(c.iloc[-1] / c.iloc[-21] - 1)
    rets60[a] = float(c.iloc[-1] / c.iloc[-61] - 1)
    rets120[a] = float(c.iloc[-1] / c.iloc[-121] - 1)
    ma20 = float(c.rolling(20).mean().iloc[-1])
    if float(c.iloc[-1]) > ma20:
        above_ma20 += 1

print("20d eqw cum: %.3f%%" % (100 * np.mean(list(rets20.values()))))
print("60d eqw cum: %.3f%%" % (100 * np.mean(list(rets60.values()))))
print("120d eqw cum: %.3f%%" % (100 * np.mean(list(rets120.values()))))
print("breadth above MA20: %d/15" % above_ma20)

# daily x-sectional dispersion (20d mean)
xs = []
for i in range(1, 21):
    day_rets = []
    for a, df in frames.items():
        if df is None or len(df) < i + 2:
            continue
        c = df["close"].astype(float)
        r = float(c.iloc[-i] / c.iloc[-i - 1] - 1)
        day_rets.append(r)
    if len(day_rets) >= 10:
        xs.append(np.std(day_rets))
print("20d mean daily x-sect dispersion: %.3f%%" % (100 * np.mean(xs)))

# vol
vols = []
for a, df in frames.items():
    if df is None or len(df) < 25:
        continue
    v = float(df["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
    vols.append(v)
print("20d ann vol mean: %.1f%% (max %.1f%%)" % (100 * np.mean(vols), 100 * np.max(vols)))

# leaders / laggards
srt = sorted(rets20.items(), key=lambda kv: -kv[1])
print("20d leaders:", ", ".join(f"{a} {100*r:+.1f}%" for a, r in srt[:5]))
print("20d laggards:", ", ".join(f"{a} {100*r:+.1f}%" for a, r in srt[-5:]))

# VIX
try:
    vix = pd.read_csv("../persistent/index_data/VIX.csv")
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix[vix["date"] <= "2035-05-03"].sort_values("date")
    v = vix["close"].astype(float)
    print("VIX last: %.1f | 20d ago: %.1f | 60d ago: %.1f" % (v.iloc[-1], v.iloc[-21], v.iloc[-61]))
except Exception as e:
    print("VIX error:", e)
