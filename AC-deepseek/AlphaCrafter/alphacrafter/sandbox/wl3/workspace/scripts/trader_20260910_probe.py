# Trader probe: account state, data flatness, ensemble, rebalance helper source
import json, math, sys, inspect
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (
    get_account_dict, get_stock_daily_data, get_index_daily_data, rebalance_to_weights,
)

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

def get_df(sym, days=300):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=days)
        return get_stock_daily_data(sym, days=days)
    except Exception:
        return None

acc = get_account_dict()
print("total_assets", round(acc.get("total_assets", 0), 2))
print("cash", round(acc.get("available_cash", 0), 2))
print("n positions", len(acc.get("positions", [])))
for p in acc.get("positions", []):
    print(" pos", p["symbol"], round(p["quantity"], 4), "mv", round(p["market_value"], 2),
          "pnl%", round(p.get("profit_loss_rate", 0) * 100, 2))

assets = list(acc.get("watch_list", []))
print("\nwatch:", assets)

print("\nasset, last_chg_date, n_flat_days, vol20, ret60, ret20")
for a in assets:
    df = get_df(a, days=300)
    if df is None or len(df) < 30:
        print(a, "NO DATA", None if df is None else len(df))
        continue
    c = df["close"].astype(float).values
    dts = pd.to_datetime(df["date"]).values
    last_chg = 0
    for i in range(len(c) - 1, 0, -1):
        if c[i] != c[i - 1]:
            last_chg = i
            break
    n_flat = len(c) - 1 - last_chg
    s = pd.Series(c)
    r = s.pct_change()
    vol20 = float(r.tail(20).std()) if len(r) >= 21 else float("nan")
    ret60 = c[-1] / c[-61] - 1 if len(c) >= 61 else float("nan")
    ret20 = c[-1] / c[-21] - 1 if len(c) >= 21 else float("nan")
    print(f"{a:10s} {str(dts[last_chg])[:10]:12s} flat={n_flat:3d} vol20={vol20*100:6.3f}% ret60={ret60*100:7.2f}% ret20={ret20*100:7.2f}%")

try:
    ens = json.load(open("factor_ensemble.json"))
    print("\nensemble method:", ens.get("method"))
    for f in ens.get("selected_factors", []):
        print(" ", f["factor_id"], "w=", round(float(f["weight"]), 4), "d=", f["direction"])
except Exception as e:
    print("ensemble err", e)

print("\n--- rebalance_to_weights source ---")
print(inspect.getsource(rebalance_to_weights))

d = json.load(open("../persistent/date.json"))
print("\ndate:", d.get("current_date"), "visible:", d.get("visible_through"), "complete:", d.get("simulation_complete"))
