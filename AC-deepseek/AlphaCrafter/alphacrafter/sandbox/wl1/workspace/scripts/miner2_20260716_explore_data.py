"""Explore data availability and formats for factor mining."""
import os
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

acc = get_account_dict()
print("watch_list from account:", acc.get("watch_list"))
print("account keys:", list(acc.keys()))

for s in WATCH:
    src = "none"
    df = None
    try:
        df = get_stock_daily_data(s, days=400)
        src = "stock"
    except Exception:
        df = None
    if df is None:
        try:
            df = get_index_daily_data(s, days=400)
            src = "index"
        except Exception as e:
            src = f"err:{e}"
    if df is not None and len(df):
        print(f"{s:10s} src={src:6s} rows={len(df):4d} first={df.iloc[0]['date']} last={df.iloc[-1]['date']} cols={list(df.columns)}")
    else:
        print(f"{s:10s} src={src:6s} NO DATA")

print("\n--- index_data files ---")
for f in sorted(os.listdir("../persistent/index_data")):
    p = os.path.join("../persistent/index_data", f)
    with open(p) as fh:
        lines = fh.readlines()
    print(f, "n_lines=", len(lines), "header=", lines[0].strip()[:120])