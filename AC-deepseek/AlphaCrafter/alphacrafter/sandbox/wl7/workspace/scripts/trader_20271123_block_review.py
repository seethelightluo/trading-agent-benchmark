"""Trader block review 2027-11-09 -> 2027-11-23: attribution, regime, holdings."""
import json
from pathlib import Path
import pandas as pd

from alphacrafter.sim.utils import (
    get_account_dict,
    get_stock_daily_data,
    get_index_daily_data,
)

acc = get_account_dict()
print("TOTAL_ASSETS", round(acc.get("total_assets", 0), 2))
print("CASH", round(acc.get("available_cash", 0), 2))
print("GROSS_POS_RATE", round(acc.get("gross_position_rate", 0), 4))
print("NET_POS_RATE", round(acc.get("net_position_rate", 0), 4))
print("PENDING_ORDERS", len(acc.get("orders", [])))

positions = {p["symbol"]: p for p in acc.get("positions", []) if p.get("quantity", 0) != 0}
print("POSITION_COUNT", len(positions))
for sym, p in sorted(positions.items()):
    mv = p.get("market_value", 0)
    print(f"  {sym}: qty={p['quantity']:.4f} mv={mv:,.0f} plr={p.get('profit_loss_rate', 0)*100:.2f}%")

# block return attribution using current holdings weight and asset block return
def get_df(sym):
    try:
        return get_stock_daily_data(sym, days=40)
    except Exception:
        return None

def get_idx(sym):
    try:
        return get_index_daily_data(sym, days=40)
    except Exception:
        return None

assets = acc.get("watch_list", [])
ret = {}
for a in assets:
    df = get_df(a) if a in positions or True else None
    if df is None:
        df = get_idx(a)
    if df is None or len(df) < 2:
        ret[a] = None
        continue
    df = df.sort_values("date")
    c = df["close"].astype(float)
    # block return from 2027-11-08 close to last close (block 11-09..11-23)
    if len(c) >= 12:
        r = c.iloc[-1] / c.iloc[-12] - 1.0
    else:
        r = c.iloc[-1] / c.iloc[0] - 1.0
    ret[a] = r

total_mv = sum(p.get("market_value", 0) for p in positions.values()) or 1.0
contrib = {}
for a, r in ret.items():
    w = positions.get(a, {}).get("market_value", 0) / total_mv if a in positions else 0.0
    contrib[a] = (w, r, w * r)

print("\nBLOCK ATTRIBUTION (2027-11-08 close -> last close):")
for a in assets:
    w, r, c = contrib.get(a, (0.0, None, 0.0))
    if r is None:
        print(f"  {a}: wt={w*100:5.2f}% ret=NA")
    else:
        print(f"  {a}: wt={w*100:5.2f}% ret={r*100:7.2f}% contrib={c*100:6.2f}pp")
print("SUM_CONTRIB_pp", round(sum(v[2] for v in contrib.values()) * 100, 2))

# regime at last close
vix = get_idx("VIX")
if vix is not None and len(vix):
    print("\nVIX_LAST", float(vix["close"].iloc[-1]))
    print("VIX_20d_ago", float(vix["close"].iloc[-12]) if len(vix) >= 12 else None)
dxy = get_idx("DXY")
if dxy is not None and len(dxy):
    print("DXY_LAST", round(float(dxy["close"].iloc[-1]), 2))

# mkt 20d trend
df_spx = get_idx("SPX")
if df_spx is not None and len(df_spx) > 25:
    c = df_spx["close"].astype(float)
    m20 = (c.iloc[-1] / c.iloc[-21] - 1.0)
    print("SPX_20d_ret", round(float(m20) * 100, 2), "%")
