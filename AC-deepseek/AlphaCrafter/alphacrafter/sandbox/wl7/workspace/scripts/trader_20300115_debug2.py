"""Trader debug 2030-01-15: inspect raw series values."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import pandas as pd

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
a = assets[0]
df = get_stock_daily_data(a, days=200)
print("columns:", list(df.columns))
print("dtypes:", df.dtypes.to_dict())
print("head:")
print(df.head(3).to_string())
print("tail:")
print(df.tail(3).to_string())
s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
print("s dtype:", s.dtype, "len:", len(s))
print("s head values:", s.head(3).tolist())
print("s iloc[:5]:", s.iloc[:5].tolist())
# check duplicates
print("index duplicated:", s.index.duplicated().sum())
print("close dtype raw:", df["close"].dtype)
print("close sample:", df["close"].iloc[:3].tolist())
print("change sample:", df["change"].iloc[:3].tolist())
print("pct_change sample:", df["pct_change"].iloc[:3].tolist())
