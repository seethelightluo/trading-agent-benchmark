from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

assets = list(get_account_dict()["watch_list"])
frames = {}
for a in assets:
    f = None
    try:
        f = get_stock_daily_data(a, days=300)
    except Exception:
        f = None
    if f is None or len(f) < 140:
        try:
            f = get_index_daily_data(a, days=300)
        except Exception:
            f = None
    if f is not None and len(f) >= 140:
        df = f.sort_values("date").copy()
        df["date"] = pd.to_datetime(df["date"])
        frames[a] = df.set_index("date")["close"].astype(float).rename(a)
panel = pd.concat(frames.values(), axis=1, join="inner")
print("panel last:", panel.index[-1])
# Decision was 2028-07-13; data visible through 2028-07-12.
panel = panel[panel.index <= pd.Timestamp("2028-07-12")]
rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
regime = "bull" if trend > 1.0 else ("bear" if trend < -1.0 else "sideways")
print(f"panel rows={len(panel)} last={panel.index[-1]}")
print(f"trend_tstat={trend:.3f} regime={regime}")
print(f"20d cross-asset mean ret={r20*100:.2f}%  std={v20*100:.2f}%")
