"""Trader block review: 2034-09-28 -> 2034-10-12 live block.

Reads account state (final weights), computes per-asset block returns from
daily data ending at the decision date, and estimates block PnL attribution
using start-of-block target weights from memory (09-28 decision).
"""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

OBS_ONLY = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

acct = get_account_dict()
nav = float(acct.get("net_assets", 0.0))
total_assets = float(acct.get("total_assets", 0.0))
avail = float(acct.get("available_cash", 0.0))
print(f"NAV={nav:,.2f} total_assets={total_assets:,.2f} cash={avail:,.2f}")
print(f"gross_pos_rate={acct.get('gross_position_rate')} net_pos_rate={acct.get('net_position_rate')}")
print(f"total_pnl={acct.get('total_profit_loss'):,.2f} rate={acct.get('total_profit_loss_rate')*100:.2f}%")

pos = {p["symbol"]: p for p in acct.get("positions", [])}
print(f"n_positions={len(pos)}")
for sym, p in sorted(pos.items()):
    print(f"  {sym}: qty={p.get('quantity'):.2f} mv={p.get('market_value'):,.0f} "
          f"w={p.get('market_value',0)/nav*100:.2f}% pnl={p.get('profit_loss'):,.0f} "
          f"({p.get('profit_loss_rate')*100:.2f}%)")

print("\n-- orders --")
for o in acct.get("orders", []):
    print(" ", o)

# per-asset block return over the 10 trading days ending 09-28 (decision date)
# get last 15 days; block approx last 10 trading days
print("\n-- per-asset 10d return ending 09-28 --")
rows = []
for sym in sorted(pos.keys()):
    try:
        df = get_stock_daily_data(sym, days=15) if sym not in OBS_ONLY else get_index_daily_data(sym, days=15)
    except Exception:
        df = None
    if df is None or len(df) < 11:
        print(f"  {sym}: no data")
        continue
    df = df.sort_values("date")
    r10 = df.iloc[-1]["close"] / df.iloc[-11]["close"] - 1.0
    rows.append((sym, r10))
    print(f"  {sym}: 10d ret={r10*100:+.2f}%")

if rows:
    print(f"\n  mean 10d ret={np.mean([r for _, r in rows])*100:+.2f}%")
