"""Trader probe: account state, rebalance helper signature, data horizon."""
import inspect, json
import pandas as pd
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
                                    get_index_daily_data, rebalance_to_weights)

acc = get_account_dict()
print("ACCOUNT KEYS:", sorted(acc.keys()))
print("total_assets:", acc.get("total_assets"), "cash:", acc.get("available_cash"))
print("watch_list:", acc.get("watch_list"))
print("positions:", json.dumps(acc.get("positions", []), indent=1)[:1500])
print("orders:", json.dumps(acc.get("orders", []), indent=1)[:800])

print("\nrebalance_to_weights signature:", inspect.signature(rebalance_to_weights))
try:
    src = inspect.getsource(rebalance_to_weights)
    print("SOURCE (first 3000 chars):\n", src[:3000])
except Exception as e:
    print("no source:", e)

print("\n--- data horizon probe ---")
for s in ["SPX", "000300.SH", "BTC", "WTI"]:
    df = get_stock_daily_data(symbol=s, days=15)
    if df is None:
        print(s, "None"); continue
    print(s, "rows:", len(df), "last date:", df["date"].iloc[-1], "prev:", df["date"].iloc[-2])
for ix in ["DXY", "VIX", "EURUSD", "SPX"]:
    df = get_index_daily_data(symbol=ix, days=15)
    if df is None:
        print("IDX", ix, "None"); continue
    print("IDX", ix, "rows:", len(df), "last date:", df["date"].iloc[-1])
