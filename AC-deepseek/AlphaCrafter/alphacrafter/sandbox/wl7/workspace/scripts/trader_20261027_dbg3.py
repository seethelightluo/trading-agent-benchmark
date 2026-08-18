import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

df = get_stock_daily_data("SPX", days=80)
s1 = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
print("exact strategy _series construction -> NaN count:", s1.isna().sum(), "of", len(s1))

# alternative: values-based
s2 = pd.Series(df["close"].astype(float).values, index=pd.to_datetime(df["date"]))
print("values-based construction -> NaN count:", s2.isna().sum(), "of", len(s2))
print("tail values:", s2.tail(3).round(2).tolist())

# check df index
print("df index type:", type(df.index), "first:", df.index[0])
print("close type:", type(df["close"]))
print("pandas version:", pd.__version__)
