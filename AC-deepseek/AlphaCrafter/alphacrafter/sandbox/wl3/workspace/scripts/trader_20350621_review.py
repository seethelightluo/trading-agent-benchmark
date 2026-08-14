"""Trader cycle review: 2035-06-07 -> 2035-06-21 block.

Inspect account state and compute per-asset block returns for attribution.
"""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

acct = get_account_dict()
print("=== ACCOUNT ===")
print("net_assets:", round(acct.get("net_assets", 0), 2))
print("total_assets:", round(acct.get("total_assets", 0), 2))
print("available_cash:", round(acct.get("available_cash", 0), 4))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("pending orders:", len(acct.get("orders", [])))
pos = {p["symbol"]: p for p in acct.get("positions", [])}
print("positions:", len(pos))

def get_df(sym, days=40):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=days)
        return get_stock_daily_data(sym, days=days)
    except Exception as e:
        print("ERR", sym, e)
        return None

print("\n=== BLOCK RETURNS (06-06 close -> last close) ===")
wl = acct.get("watch_list", [])
block_start = "2035-06-06"
tot_nav = float(acct.get("net_assets", 0))
contrib = 0.0
for sym in wl:
    df = get_df(sym, days=30)
    if df is None or len(df) < 3:
        print(sym, "no data")
        continue
    df = df.sort_values("date").reset_index(drop=True)
    s = df["date"].astype(str)
    # close on/after block_start (first close at or after 06-06) and last close
    idx_start = s[s <= block_start].index.max()
    import math
    if idx_start is None or (isinstance(idx_start, float) and math.isnan(idx_start)):
        print(sym, "no start date")
        continue
    p0 = float(df.loc[idx_start, "close"])
    p1 = float(df.iloc[-1]["close"])
    r = p1 / p0 - 1.0
    w = pos.get(sym, {}).get("market_value", 0.0) / tot_nav if tot_nav else 0.0
    contrib += w * r
    print(f"{sym:10s} p0={p0:10.4f} p1={p1:10.4f} ret={r*100:7.2f}%  w_cur={w*100:5.2f}%  contrib={w*r*100:6.2f}pp")
print(f"approx total contrib (current weights): {contrib*100:.2f}pp")
