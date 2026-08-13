"""Trader block analysis for block 2033-06-17 -> 2033-07-01."""
import json
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
D0, D1 = "2033-06-17", "2033-07-01"

with open("../persistent/account.json") as f:
    acc = json.load(f)
tot0 = acc.get("total_assets", 0)  # end-of-block total (0701)
# reconstruct start-of-block total from last summary + pnl? use account snapshot at 0617 is unavailable;
# step period return +1.15% -> start ~ 1,187,177/1.0115

print("=== per-asset block return % (close D0 -> close D1) ===")
ret = {}
for a in WATCH:
    try:
        df = get_stock_daily_data(symbol=a, days=60)
        if df is None or len(df) == 0:
            print(f"{a}: NO DATA")
            continue
        df = df.copy()
        df["date"] = df["date"].astype(str)
        sub = df[(df["date"] >= D0) & (df["date"] <= D1)]
        if len(sub) < 2:
            print(f"{a}: insufficient rows {len(sub)}")
            continue
        c0 = float(sub.iloc[0]["close"])
        c1 = float(sub.iloc[-1]["close"])
        r = (c1 / c0 - 1.0) * 100.0
        ret[a] = r
        print(f"{a}: {r:+.2f}%  (c0 {c0:.4f} -> c1 {c1:.4f})")
    except Exception as e:
        print(f"{a}: ERR {e}")

# executed weights at 0617 (cost-basis market value at 0701 / total)
print("\n=== executed weights @0617 (est from 0701 market values / total) ===")
tot = acc.get("total_assets", 1.0)
for p in sorted(acc.get("positions", []), key=lambda x: -x.get("market_value", 0)):
    w = p.get("market_value", 0) / tot * 100
    print(f"  {p['symbol']}: {w:.2f}%")

print("\n=== account ===")
print("total_assets 0701:", acc.get("total_assets"))
print("cash:", acc.get("available_cash"))
print("last_rebalance_date:", acc.get("last_rebalance_date"))
print("orders pending:", len(acc.get("orders", [])))
