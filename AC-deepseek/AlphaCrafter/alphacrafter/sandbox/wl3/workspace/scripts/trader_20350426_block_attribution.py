import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

assets = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
rows = []
for a in assets:
    try:
        df = get_stock_daily_data(a, days=40)
    except Exception:
        df = get_index_daily_data(a, days=40)
    df = df.sort_values("date").reset_index(drop=True)
    dates = [str(x)[:10] for x in df["date"]]
    if "2035-04-11" not in dates or "2035-04-25" not in dates:
        print(a, "missing anchor", dates[-5:])
        continue
    c0 = df.loc[dates.index("2035-04-11"), "close"]
    c1 = df.loc[dates.index("2035-04-25"), "close"]
    rows.append((a, c1/c0 - 1.0))
rows.sort(key=lambda x: -x[1])
for a, r in rows:
    print(f"{a:10s} {r*100:7.2f}%")
