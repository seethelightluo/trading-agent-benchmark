from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
import pandas as pd
import numpy as np

acc = get_account_dict()
wl = acc['watch_list']
print("WATCHLIST:", wl, "N=", len(wl))
for sym in wl:
    df = get_stock_daily_data(symbol=sym, days=3200)
    if df is None:
        print(sym, "NO DATA")
        continue
    print(sym, "rows=", len(df), "from", df.date.min().date(), "to", df.date.max().date(),
          "cols=", list(df.columns))