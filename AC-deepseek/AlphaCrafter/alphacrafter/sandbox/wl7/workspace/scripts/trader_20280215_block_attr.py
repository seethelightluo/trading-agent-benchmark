"""Block attribution for 2028-02-01 -> 2028-02-15 step (trader cycle)."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd

acc = get_account_dict()
assets = acc.get("watch_list", [])
pos = {p["symbol"]: p for p in acc.get("positions", [])}
total = acc.get("total_assets", 0.0)

rows = []
for a in assets:
    df = get_stock_daily_data(a, days=200)
    if df is None or len(df) < 20:
        df = get_index_daily_data(a, days=200)
    if df is None or len(df) < 20:
        print(a, "NO DATA")
        continue
    df = df.sort_values("date")
    c = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
    w = c.tail(11)  # 10 trading days + anchor
    ret = w.iloc[-1] / w.iloc[0] - 1.0
    wt = pos[a]["market_value"] / total if a in pos else 0.0
    rows.append((a, ret, wt, ret * wt))

rows.sort(key=lambda r: r[3], reverse=True)
print(f"{'asset':10s} {'ret%':>8s} {'wt%':>6s} {'contrib%':>9s}")
for a, r, wt, contrib in rows:
    print(f"{a:10s} {r*100:8.2f} {wt*100:6.2f} {contrib*100:9.3f}")
print("sum contrib%:", round(sum(r[3] for r in rows) * 100, 3))

# defensive floor weight snapshot
def_wt = sum(pos[a]["market_value"] / total for a in ("XAU", "US10Y", "CN10Y") if a in pos)
print("defensive floor wt%:", round(def_wt * 100, 2))
