import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = list(get_account_dict()["watch_list"])
frames = {}
for a in assets:
    f = get_stock_daily_data(a, days=60)
    if f is None or len(f) < 40:
        f = get_index_daily_data(a, days=60)
    if f is not None and "close" in f:
        frames[a] = f["close"].astype(float).rename(a)

panel = pd.concat(frames, axis=1, join="inner").dropna()
rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
regime = "bull" if trend > 1.0 else ("bear" if trend < -1.0 else "sideways")
print("last index", panel.index[-1], "trend", round(trend, 3), "regime", regime)

r20a = (panel.iloc[-1] / panel.iloc[-21] - 1.0).sort_values()
for a, v in r20a.items():
    print(f"{a:8s} {v:8.2%}")
