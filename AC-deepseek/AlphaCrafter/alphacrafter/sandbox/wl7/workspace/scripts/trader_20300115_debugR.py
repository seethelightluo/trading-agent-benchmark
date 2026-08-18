"""Trader debug 2030-01-15: why does R become empty?"""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import pandas as pd

acc = get_account_dict()
assets = list(acc.get("watch_list", []))

series = {}
for a in assets:
    df = get_stock_daily_data(a, days=200)
    s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
    series[a] = s

usable = {a: s.pct_change().rename(a) for a, s in series.items()}
print("usable count:", len(usable))
print("first series head:")
print(usable[assets[0]].head(3))
print("first series non-null count:", usable[assets[0]].notna().sum())

R_raw = pd.concat(usable, axis=1, join="inner")
print("R_raw shape:", R_raw.shape)
print("R_raw NaN per col (first 3):")
print(R_raw.isna().sum().head(3))
print("R_raw NaN total:", int(R_raw.isna().sum().sum()))

R_drop = R_raw.dropna()
print("R_drop shape:", R_drop.shape)
