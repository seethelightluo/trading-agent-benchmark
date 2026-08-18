import pandas as pd, os
df = pd.read_csv('../persistent/stock_data/SPX.csv')
print(df.columns.tolist())
print(df.head(3))
print(df.tail(3))
print("rows:", len(df))
ix = pd.read_csv('../persistent/index_data/VIX.csv')
print(ix.columns.tolist())
print(ix.tail(3))