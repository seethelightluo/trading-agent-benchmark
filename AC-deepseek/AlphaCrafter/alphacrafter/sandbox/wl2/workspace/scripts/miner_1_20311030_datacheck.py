"""miner_1 2031-10-30: data availability check through visible date 2031-10-29."""
import sys
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))
VISIBLE = "2031-10-29"

def load_asset(sym, days=4500):
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
    if df is None or len(df) == 0:
        print(f"{s:10s} NO DATA")
        continue
    last = df.index[-1]
    last_close = df["close"].iloc[-1]
    print(f"{s:10s} rows={len(df):5d} last={last} last_close={last_close:.4f} at_visible={last <= VISIBLE}")
