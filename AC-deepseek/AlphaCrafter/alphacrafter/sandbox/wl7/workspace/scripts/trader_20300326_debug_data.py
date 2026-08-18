import json, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

with open("../persistent/date.json") as f:
    print("date.json:", json.load(f))

df = get_stock_daily_data("SPX", days=80)
print("SPX type:", type(df))
if df is not None:
    print("SPX shape:", df.shape)
    print(df.tail(5).to_string())
    print("date range:", df["date"].min(), "->", df["date"].max())
    print("close tail:", df["close"].tail(5).tolist())