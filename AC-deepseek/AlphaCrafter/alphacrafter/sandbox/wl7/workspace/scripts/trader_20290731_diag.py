"""Trader diagnostic: account state + regime snapshot at 2029-07-31 decision."""
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
print("assets:", assets)
print("n assets:", len(assets))
print("total_assets:", acc.get("total_assets"))
print("net_assets:", acc.get("net_assets"))
print("available_cash:", acc.get("available_cash"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("positions:")
for p in acc.get("positions", []):
    print("  ", p["symbol"], p["direction"], "qty", round(p["quantity"], 4),
          "mv", round(p["market_value"], 2), "pl", round(p["profit_loss"], 2))
print("orders:", acc.get("orders"))

# ensemble check
try:
    ens = json.loads(Path("factor_ensemble.json").read_text())
    sel = [(it["factor_id"], it["weight"], it.get("direction", 1))
           for it in ens.get("selected_factors", [])]
    print("ensemble selected_factors:")
    for s in sel:
        print("  ", s)
except Exception as e:
    print("ensemble err:", e)

# regime snapshot
for sym in ["VIX", "DXY", "USDJPY", "EURUSD", "USDCNY"]:
    try:
        df = get_index_daily_data(sym, days=30)
        if df is not None and len(df) >= 2:
            c = df["close"].astype(float)
            print(f"{sym}: last {c.iloc[-1]:.2f}  10d {(c.iloc[-1]/c.iloc[-11]-1)*100:+.2f}%  20d {(c.iloc[-1]/c.iloc[-21]-1)*100:+.2f}%" if len(c) >= 21 else
                  f"{sym}: last {c.iloc[-1]:.2f}  10d {(c.iloc[-1]/c.iloc[-11]-1)*100:+.2f}%")
    except Exception as e:
        print(sym, "err", e)

# last data date for watchlist
for sym in assets[:6]:
    try:
        df = get_stock_daily_data(sym, days=10)
        print(sym, "last date:", df["date"].iloc[-1] if df is not None and len(df) else None)
    except Exception as e:
        print(sym, "err", e)
