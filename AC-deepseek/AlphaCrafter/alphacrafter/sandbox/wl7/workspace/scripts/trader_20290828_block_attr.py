"""Block attribution for 2029-08-14 -> 2029-08-28 (10-trading-day cycle).

Uses pre-step account.json.bak as start snapshot (quantities unchanged, no
rebalance executed) and daily closes through 2029-08-28.
"""
import json
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

BAK = "../persistent/account.json.bak"
CUR = "../persistent/account.json"

bak = json.load(open(BAK))
cur = json.load(open(CUR))

def posmap(a):
    return {p["symbol"]: p for p in a.get("positions", [])}

bp, cp = posmap(bak), posmap(cur)
qty_same = all(abs(bp[s]["quantity"] - cp[s]["quantity"]) < 1e-9 for s in bp)
print("quantities unchanged (no rebalance):", qty_same)
print("start net_assets:", round(bak["net_assets"], 2), "end net_assets:", round(cur["net_assets"], 2),
      "block PnL:", round(cur["net_assets"] - bak["net_assets"], 2),
      "ret:", round((cur["net_assets"] / bak["net_assets"] - 1) * 100, 3), "%")

rows = []
for sym, p in bp.items():
    try:
        df = get_stock_daily_data(sym, days=40)
    except Exception:
        df = None
    if df is None or len(df) < 15:
        try:
            df = get_index_daily_data(sym, days=40)
        except Exception:
            df = None
    if df is None:
        continue
    df = df.sort_values("date").reset_index(drop=True)
    dts = pd.to_datetime(df["date"])
    end_mask = dts <= pd.Timestamp("2029-08-28")
    if not end_mask.any():
        continue
    end_i = dts[end_mask].idxmax()
    start_i = (dts <= pd.Timestamp("2029-08-14")).idxmax()
    p_start = float(df.loc[start_i, "close"])
    p_end = float(df.loc[end_i, "close"])
    ret = p_end / p_start - 1.0
    mv_start = p.get("market_value", 0.0)
    wt = mv_start / bak["net_assets"]
    rows.append((sym, wt, ret, wt * ret, mv_start))

rows.sort(key=lambda r: -abs(r[3]))
print(f"\n{'sym':9s} {'wt%':>6s} {'ret%':>8s} {'contrib_pp':>9s}")
tot = 0.0
for sym, wt, ret, contrib, mv in rows:
    print(f"{sym:9s} {wt*100:6.2f} {ret*100:8.2f} {contrib*100:9.2f}")
    tot += contrib
print(f"\nsum contrib {tot*100:.2f} pp (vs actual {(cur['net_assets']/bak['net_assets']-1)*100:.2f} pp)")
