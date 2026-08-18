"""Compute regime at the 2028-05-04 decision date (data visible through 2028-05-03)."""
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = list(get_account_dict()["watch_list"])
frames = {}
for a in assets:
    try:
        f = get_stock_daily_data(a, days=300)
    except Exception:
        f = None
    frames[a] = f

closes = {a: (f.close.astype(float) if f is not None else None) for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
panel = pd.concat(usable, axis=1, join="inner")
# Restrict to data visible at decision date (through 2028-05-03)
panel = panel[panel.index <= "2028-05-03"]
rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
regime = "bull" if trend > 1.0 else ("bear" if trend < -1.0 else "sideways")
print("decision_date_last_bar:", panel.index[-1])
print("trend_tstat:", round(trend, 3))
print("regime:", regime)
print("panel cols:", list(panel.columns))
