"""miner_1 2026-08-27 probe: data availability and ranges."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

DATE_PATH = "../persistent/date.json"
date_state = json.load(open(DATE_PATH))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
print("GRID size:", len(GRID), "first:", GRID[0], "last:", GRID[-1])

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))

def load_asset(sym, days=2100):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

for s in ASSETS:
    df = load_asset(s)
    if df is None:
        print(f"{s}: NO DATA")
        continue
    print(f"{s}: rows={len(df)} first={df.index[0]} last={df.index[-1]} "
          f"in_grid={df.index[-1] in GRID} cols={list(df.columns)}")
    # check overlap with GRID
    common = df.index.intersection(GRID)
    print(f"   overlap_with_grid={len(common)} last_common={common[-1] if len(common) else 'NONE'}")

# macro signals
import os
for name in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    p = f"../persistent/index_data/{name}.csv"
    if os.path.exists(p):
        m = pd.read_csv(p)
        m["date"] = pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d")
        m = m.set_index("date")
        print(f"macro {name}: rows={len(m)} last={m.index[-1]} cols={list(m.columns)[:6]}")
