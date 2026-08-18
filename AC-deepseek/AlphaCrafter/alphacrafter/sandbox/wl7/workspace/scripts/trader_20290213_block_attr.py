"""Block attribution for 2029-02-13 -> 2029-02-27 step (decision 02-13)."""
import json
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
pos = {p["symbol"]: p for p in acc.get("positions", [])}
nav = acc.get("net_assets", 0.0)

START = pd.Timestamp("2029-02-12")  # last completed day before decision
END = pd.Timestamp("2029-02-27")    # step end

rows = []
for a in assets:
    df = get_stock_daily_data(a, days=40) or get_index_daily_data(a, days=40)
    if df is None or len(df) == 0:
        rows.append((a, None, None, None, pos.get(a, {}).get("market_value", 0.0) / nav))
        continue
    df = df.sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    pre = df[df["date"] <= START]
    post = df[df["date"] <= END]
    if len(pre) == 0 or len(post) == 0:
        rows.append((a, None, None, None, pos.get(a, {}).get("market_value", 0.0) / nav))
        continue
    p0 = pre.iloc[-1]["close"]
    p1 = post.iloc[-1]["close"]
    r = p1 / p0 - 1.0
    w = pos.get(a, {}).get("market_value", 0.0) / nav
    rows.append((a, r, w, r * w, w))

total_attr = 0.0
print(f"{'asset':10s} {'ret%':>8s} {'wt%':>7s} {'contrib%':>9s}")
for a, r, w, c, _ in sorted(rows, key=lambda x: -(x[3] if x[3] is not None else -99)):
    if r is None:
        print(f"{a:10s} {'NA':>8s} {w*100:7.2f} {'--':>9s}")
        continue
    total_attr += c
    print(f"{a:10s} {r*100:8.2f} {w*100:7.2f} {c*100:9.2f}")
print(f"{'TOTAL':10s} {'':>8s} {'':>7s} {total_attr*100:9.2f}")
print("nav", round(nav, 2))
