"""Compute per-asset block returns for 2032-10-14 -> 2032-10-28 cycle.

Data API returns data through the previous completed trading day (10-27);
use 10-14 close as block-start reference (last visible at rebalance+1).
"""
import json

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

def get_df(symbol, days=40):
    try:
        if symbol in OBS:
            return get_index_daily_data(symbol, days=days)
        return get_stock_daily_data(symbol, days=days)
    except Exception:
        return None

a = json.load(open("../persistent/account.json"))
pos = {p["symbol"]: p for p in a.get("positions", [])}
nav = float(a.get("net_assets", 0))

rows = []
for sym in pos:
    df = get_df(sym, days=40)
    if df is None or len(df) < 12:
        continue
    df = df.sort_values("date").reset_index(drop=True)
    d14 = df[df["date"].astype(str).str.startswith("2032-10-14")]
    if len(d14) == 0:
        continue
    c14 = float(d14.iloc[0]["close"])
    c_last = float(df.iloc[-1]["close"])  # 2032-10-27
    r = c_last / c14 - 1.0
    mv = float(pos[sym]["market_value"])
    # contribution approx using start value = mv/(1+r)
    contrib = mv * r / (1 + r) if abs(1 + r) > 1e-9 else 0.0
    rows.append((sym, r * 100, contrib, mv / nav * 100))

rows.sort(key=lambda x: -x[1])
print(f"{'sym':10s} {'ret%':>8s} {'contrib':>10s} {'w%':>6s}")
for sym, r, c, w in rows:
    print(f"{sym:10s} {r:8.2f} {c:10.0f} {w:6.2f}")
tot_contrib = sum(r[2] for r in rows)
print(f"\nsum contrib ~ {tot_contrib:,.0f} on nav {nav:,.0f} = {tot_contrib/nav*100:.2f}%")
