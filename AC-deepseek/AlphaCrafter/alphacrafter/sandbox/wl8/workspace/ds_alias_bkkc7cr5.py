import pandas as pd, os
df = pd.read_csv('../persistent/stock_data/SPX.csv')
print("SPX rows:", len(df), "first:", df['date'].iloc[0], "last:", df['date'].iloc[-1])
df2 = pd.read_csv('../persistent/stock_data/000300.SH.csv')
print("000300 rows:", len(df2), "first:", df2['date'].iloc[0], "last:", df2['date'].iloc[-1])
for f in ['VIX','USDCNY','DXY']:
    d = pd.read_csv(f'../persistent/index_data/{f}.csv')
    print(f, "rows:", len(d), "first:", d['date'].iloc[0], "last:", d['date'].iloc[-1])
print(os.listdir('../persistent/'), os.listdir('../persistent/stock_data/')[:5])