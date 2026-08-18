"""miner_1: check data availability via simulator APIs (date-gated to 2031-07-24)."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd

acc = get_account_dict()
print("watch_list:", acc.get("watch_list"))

syms = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
        "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

for s in syms:
    df = get_index_daily_data(symbol=s, days=4000)
    if df is None:
        df = get_stock_daily_data(symbol=s, days=4000)
    if df is None:
        print(s, "NO DATA")
        continue
    print(s, "rows:", len(df), "range:", df["date"].iloc[0].date(), "->", df["date"].iloc[-1].date(),
          "cols:", df.columns.tolist())
