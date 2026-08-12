"""miner_1 2028-03-23 data probe: check volume/flat-feed quality per asset."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))

for s in ASSETS:
    df = get_stock_daily_data(s, days=2400)
    if df is None:
        print(f"{s:10s} NO DATA")
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    last = df.index[-1]
    n = len(df)
    vol = df["volume"]
    flat_vol = (vol.diff().abs() < 1e-9).mean() if vol.notna().sum() > 10 else float('nan')
    vol_na = vol.isna().mean()
    close = df["close"]
    flat_close = (close.diff().abs() < 1e-12).mean()
    print(f"{s:10s} n={n:5d} last={last} vol_na={vol_na:.2f} flat_vol={flat_vol:.2f} flat_close={flat_close:.3f}")
