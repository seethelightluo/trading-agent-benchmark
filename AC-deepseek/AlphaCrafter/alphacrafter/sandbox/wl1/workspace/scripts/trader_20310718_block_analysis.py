"""Trader block analysis 2031-07-04 -> 2031-07-18 (10 trading days).

Compute per-asset block returns and estimated PnL contributions using
holdings frozen at the 2025-04-25 execution (unchanged through this block).
"""
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

START, END = "2031-07-04", "2031-07-18"
WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

acc = get_account_dict()
nav_end = acc["net_assets"]
pos = {p["symbol"]: p for p in acc["positions"]}

rows = []
nav_start = 0.0
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=400)
    if df is None:
        print(a, "NO DATA")
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    c_start = float(df.loc[df.index <= pd.Timestamp(START), "close"].iloc[-1])
    c_end = float(df.loc[df.index <= pd.Timestamp(END), "close"].iloc[-1])
    ret = c_end / c_start - 1.0
    qty = pos.get(a, {}).get("quantity", 0.0)
    mv_start = qty * c_start
    nav_start += mv_start
    rows.append((a, ret, mv_start))

print(f"NAV end: {nav_end:,.2f}  NAV start(est via holdings): {nav_start:,.2f}")
for a, ret, mv in rows:
    w0 = mv / nav_start if nav_start else 0
    contrib = w0 * ret
    print(f"{a:10s} ret {ret*100:8.2f}%  w0 {w0*100:6.2f}%  contrib {contrib*100:7.2f}pp")

tot_contrib = sum(w0 * ret for _, ret, mv in rows for w0 in [mv / nav_start])
print(f"sum contrib: {tot_contrib*100:.2f}pp")
