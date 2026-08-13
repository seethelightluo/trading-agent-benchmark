"""Check whether the 2031-10-06 rebalance executed: compare cost prices vs closes."""
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data

def near(df, d):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    w = df[(df["date"] >= pd.Timestamp(d) - pd.Timedelta(days=4)) &
           (df["date"] <= pd.Timestamp(d) + pd.Timedelta(days=4))]
    return w

for sym in ["SX5E", "XAU", "COPPER", "N225", "WTI", "SPX", "000688.SH", "NDX", "US10Y", "SOX"]:
    df = get_index_daily_data(sym, days=250)
    if df is None:
        print(sym, "NO DATA")
        continue
    a = near(df, "2031-06-13")
    b = near(df, "2031-10-03")
    ca = a["close"].iloc[-1] if len(a) else float("nan")
    cb = b["close"].iloc[-1] if len(b) else float("nan")
    print(f"{sym:10s} 06-13 close={ca:12.2f}  10-03 close={cb:12.2f}")
