"""Trader market-state dump as of 2026-11-05 (data through previous close)."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (
    get_account_dict,
    get_stock_daily_data,
    get_index_daily_data,
)

acc = get_account_dict()
print("=== ACCOUNT ===")
print("total_assets", round(acc.get("total_assets", 0), 2))
print("net_assets", round(acc.get("net_assets", 0), 2))
print("available_cash", round(acc.get("available_cash", 0), 2))
print("gross_position_rate", acc.get("gross_position_rate"))
print("net_position_rate", acc.get("net_position_rate"))
print("watch_list", acc.get("watch_list"))
print("positions:")
for p in acc.get("positions", []):
    print(" ", p["symbol"], p["direction"], round(p.get("quantity", 0), 4),
          "px", round(p.get("current_price", 0), 4),
          "mv", round(p.get("market_value", 0), 2),
          "pnl%", round(p.get("profit_loss_rate", 0) * 100, 2))
print("orders:", acc.get("orders", []))

wl = acc.get("watch_list", [])
OBS = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]


def series(symbol, days=80, index=False):
    try:
        df = (get_index_daily_data(symbol, days=days) if index
              else get_stock_daily_data(symbol, days=days))
    except Exception as e:
        return None
    if df is None or "close" not in df.columns or len(df) < 5:
        return None
    df = df.sort_values("date")
    return pd.Series(df["close"].astype(float).values, index=pd.DatetimeIndex(df["date"]))


print("\n=== ASSET STATE (as of last close) ===")
rows = []
for a in wl + OBS:
    idx = a in OBS
    c = series(a, 80, index=idx)
    if c is None:
        rows.append((a, None, None, None, None, None, None, None))
        continue
    r = c.pct_change()
    ret5 = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 6 else None
    ret10 = float(c.iloc[-1] / c.iloc[-11] - 1) if len(c) > 11 else None
    ret21 = float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) > 22 else None
    ret40 = float(c.iloc[-1] / c.iloc[-41] - 1) if len(c) > 41 else None
    vol20 = float(r.tail(20).std()) if len(r) >= 20 else None
    rows.append((a, c.iloc[-1], ret5, ret10, ret21, ret40, vol20, c.index[-1].date()))

hdr = f"{'sym':8s} {'last':>10s} {'r5':>7s} {'r10':>7s} {'r21':>7s} {'r40':>7s} {'vol20':>7s} date"
print(hdr)
for a, last, r5, r10, r21, r40, v20, dt in rows:
    if last is None:
        print(f"{a:8s} NO DATA")
        continue
    f = lambda x: f"{x*100:6.1f}%" if x is not None else "   n/a"
    print(f"{a:8s} {last:10.2f} {f(r5)} {f(r10)} {f(r21)} {f(r40)} {v20*100:6.1f}% {dt}")