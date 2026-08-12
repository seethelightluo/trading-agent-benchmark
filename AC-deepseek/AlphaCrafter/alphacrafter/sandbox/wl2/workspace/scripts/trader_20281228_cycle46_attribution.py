"""Cycle46 attribution: block 2028-12-14 -> 2028-12-28.
Uses account.json (post-step, 12-28) vs account.json.bak (pre-step, 12-14)
and price data visible through the simulator API to compute per-asset returns
and approximate contribution."""
import json
from pathlib import Path

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

BASE = Path(__file__).resolve().parent.parent
with open(BASE / "../persistent/account.json") as f:
    cur = json.load(f)
with open(BASE / "../persistent/account.json.bak") as f:
    bak = json.load(f)

assets = [p["symbol"] for p in cur["positions"]]
na_cur = float(cur["net_assets"])
na_bak = float(bak["net_assets"])
print(f"NAV bak (12-14): {na_bak:,.2f}")
print(f"NAV cur (12-28): {na_cur:,.2f}")
print(f"Block return: {(na_cur/na_bak - 1)*100:.3f}%")

q_bak = {p["symbol"]: float(p["quantity"]) for p in bak["positions"]}
q_cur = {p["symbol"]: float(p["quantity"]) for p in cur["positions"]}
mv_bak = {p["symbol"]: float(p["market_value"]) for p in bak["positions"]}
mv_cur = {p["symbol"]: float(p["market_value"]) for p in cur["positions"]}

# price return over the block from raw data (close 12-14 vs close 12-28)
print("\nPer-asset block returns & weight drift (12-14 -> 12-28):")
rets = {}
for a in assets:
    df = None
    try:
        df = get_stock_daily_data(a, days=40)
    except Exception:
        pass
    if df is None or len(df) < 5:
        try:
            df = get_index_daily_data(a, days=40)
        except Exception:
            pass
    if df is None or len(df) < 5:
        print(f"  {a}: no data")
        continue
    df = df.sort_values("date")
    c_old = float(df.iloc[-11]["close"])  # ~12-14 close
    c_new = float(df.iloc[-1]["close"])   # 12-28 close
    r = c_new / c_old - 1.0
    rets[a] = r
    w_bak = mv_bak[a] / na_bak
    w_cur = mv_cur[a] / na_cur
    contrib = w_bak * r
    action = ""
    if q_cur[a] > q_bak[a] * 1.01:
        action = "BUY/ADD"
    elif q_cur[a] < q_bak[a] * 0.99:
        action = "SELL/TRIM"
    else:
        action = "hold"
    print(f"  {a:10s} r={r*100:7.2f}%  w {w_bak*100:5.2f}->{w_cur*100:5.2f}  contrib~{contrib*100:+6.2f}pp  {action}")

tot = sum(rets.get(a, 0.0) * mv_bak[a] / na_bak for a in assets)
print(f"\nSum of w*r (buy-and-hold approx): {tot*100:+.2f}%")
