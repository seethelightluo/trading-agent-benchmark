"""Trader block review 2034-12-07 -> 2034-12-21: per-asset returns & attribution."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def get_df(sym, days=40):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=days)
        return get_stock_daily_data(sym, days=days)
    except Exception:
        return None

a = json.load(open("../persistent/account.json"))
nav_pre = 1593655.179815562
nav_post = a["net_assets"]
print(f"nav {nav_pre:.2f} -> {nav_post:.2f}  pnl={nav_post - nav_pre:.2f} "
      f"({(nav_post / nav_pre - 1) * 100:.3f}%)")

rh = a.get("rebalance_history") or []
tgt = None
for r in rh:
    if r.get("date") == "2034-12-07":
        tgt = r.get("target_weights")
        print(f"rebalance 12-07: transferred {r['transferred_notional']:.0f} "
              f"cost {r['cost']:.2f} ({r['cost_bps']}bps)")
        break
if tgt is None:
    print("no 12-07 target found")
    tgt = {s: 1.0 / 15 for s in WATCH}

rows = []
for sym in WATCH:
    df = get_df(sym, days=30)
    if df is None or len(df) < 2:
        print(sym, "no data")
        continue
    df = df.sort_values("date")
    dts = df["date"].astype(str).tolist()
    p0 = p1 = None
    for i in range(len(df) - 1, -1, -1):
        if dts[i] <= "2034-12-07" and p0 is None:
            p0 = float(df["close"].iloc[i])
        if dts[i] <= "2034-12-21" and p1 is None:
            p1 = float(df["close"].iloc[i])
        if p0 is not None and p1 is not None:
            break
    if p0 is None:
        p0 = float(df["close"].iloc[0])
    if p1 is None:
        p1 = float(df["close"].iloc[-1])
    r = p1 / p0 - 1.0
    rows.append((sym, r, tgt.get(sym, 0.0)))

rows.sort(key=lambda x: -x[1])
tot = 0.0
for sym, r, w in rows:
    contrib = w * r
    tot += contrib
    print(f"{sym:10s} w={w * 100:5.2f}%  ret={r * 100:7.2f}%  contrib={contrib * 100:6.3f}pp")
print(f"sum contrib ~ {tot * 100:.2f}%")
