"""Check when flat artifacts started for SX5E/BTC/US10Y/CN10Y and general data health."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

date_state = json.load(open("../persistent/date.json"))
TRADING_DAYS = date_state["trading_days"]

for s in ["SX5E", "BTC", "US10Y", "CN10Y", "ETH"]:
    df = get_stock_daily_data(s, days=2500)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    close = df["close"].astype(float)
    diff = close.diff().abs()
    flat = diff < 1e-12
    # find last contiguous flat run
    runs = []
    start = None
    for i in range(len(flat)):
        if flat.iloc[i]:
            if start is None:
                start = flat.index[i]
        else:
            if start is not None:
                runs.append((start, flat.index[i - 1]))
                start = None
    if start is not None:
        runs.append((start, flat.index[-1]))
    last_flat_run = runs[-1] if runs else None
    print("%-8s last_flat_run=%s  (total flat days=%d)" % (s, last_flat_run, int(flat.sum())))
