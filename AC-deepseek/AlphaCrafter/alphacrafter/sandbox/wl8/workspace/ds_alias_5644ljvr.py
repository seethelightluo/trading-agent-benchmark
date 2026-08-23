import pandas as pd
df = pd.read_csv('../persistent/stock_data/SPX.csv')
print(df.head(3))
print(df.tail(3))
print(df.columns.tolist())