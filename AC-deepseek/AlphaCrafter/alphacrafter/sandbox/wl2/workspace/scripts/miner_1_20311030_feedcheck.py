"""miner_1 2031-10-30: check whether flat-feed names now have real (non-zero) returns."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU",
          "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
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
    if df is None:
        print(s, "NO DATA")
        continue
    c = df["close"].astype(float)
    ret = c.pct_change()
    tail = ret.dropna().tail(60)
    nz = (tail.abs() > 1e-9).sum()
    print(f"{s:10s} last60 non-zero rets: {nz:3d}/60   last20 mean ret: {ret.dropna().tail(20).mean()*100:+.3f}%  "
          f"20d ret: {(c.iloc[-1]/c.iloc[-21]-1)*100:+.2f}%  60d ret: {(c.iloc[-1]/c.iloc[-61]-1)*100:+.2f}%")
