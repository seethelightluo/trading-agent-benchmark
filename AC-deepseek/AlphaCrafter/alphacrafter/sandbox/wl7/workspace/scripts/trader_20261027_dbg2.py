import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

df = get_stock_daily_data("SPX", days=40)
print("rows", len(df))
print(df[["date", "close"]].tail(5).to_string())
print("close NaN count:", df["close"].isna().sum(), "of", len(df))
s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
print("series tail:")
print(s.tail(3))
print("last:", s.iloc[-1], "type", type(s.iloc[-1]))
