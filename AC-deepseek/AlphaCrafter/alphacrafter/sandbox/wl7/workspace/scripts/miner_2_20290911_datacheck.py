"""miner_2 datacheck: verify high/low/volume availability + recent close tail."""
import json
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DATA_DIR = "../persistent/stock_data"

d = json.load(open("../persistent/date.json"))
vis = d["visible_through"]
print("visible_through:", vis)

cal = pd.DatetimeIndex(pd.to_datetime([x for x in d["trading_days"] if x <= vis]))
print("master calendar n:", len(cal), "last:", cal[-1])

for a in ASSETS:
    df = pd.read_csv(f"{DATA_DIR}/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date").reindex(cal)
    n_hl = s["high"].notna().mean()
    n_vol = s["volume"].notna().mean()
    last = s["close"].dropna().iloc[-1] if s["close"].notna().any() else np.nan
    last_date = s.index[s["close"].notna()][-1] if s["close"].notna().any() else None
    print(f"{a:12s} high_cov={n_hl:.2f} vol_cov={n_vol:.2f} last_close={last:.4f} last_date={last_date.date() if last_date is not None else None}")
