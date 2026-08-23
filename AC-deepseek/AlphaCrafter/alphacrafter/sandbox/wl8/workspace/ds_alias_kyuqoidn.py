import pandas as pd
df = pd.read_csv('../persistent/stock_data/SPX.csv')
print("columns:", list(df.columns))
d = pd.read_csv('../persistent/stock_data/000688.SH.csv')
print("000688 cols:", list(d.columns))
print(d.tail(5).to_string())
print("---NDX flat check---")
for s in ['000688.SH','SOX','NDX','CN10Y']:
    d = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    d = d[d['date']<='2030-04-17']
    print(s, "n_dates:", len(d), "close nunique:", d['close'].nunique(), "last close:", float(d['close'].iloc[-1]))