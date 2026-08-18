"""Compute regime at 2027-11-04 block end for the memory log."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import pandas as pd

assets = get_account_dict()["watch_list"]
frames = {}
for a in assets:
    try:
        df = get_stock_daily_data(a, days=60)
        if df is not None and len(df) >= 30:
            frames[a] = df.set_index("date")["close"].astype(float)
    except Exception:
        pass
panel = pd.concat(frames, axis=1, join="inner")
rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
regime = "bull" if trend > 1.0 else ("bear" if trend < -1.0 else "sideways")
print("trend_tstat", round(trend, 3), "regime", regime)
print("mkt_20d_mean_ret", round(r20 * 100, 3), "pct")
