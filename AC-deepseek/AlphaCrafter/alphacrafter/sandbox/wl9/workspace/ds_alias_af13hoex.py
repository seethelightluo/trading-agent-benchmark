import pandas as pd
df=pd.read_csv('../persistent/stock_data/SPX.csv')
print(df.columns.tolist())
print(df.head(3))
print(df.tail(3))
print("rows",len(df))
print("---INDEX DATA---")
for f in ['VIX','DXY','USDCNY']:
    d=pd.read_csv('../persistent/index_data/'+f+'.csv')
    print(f,d.columns.tolist(),len(d),d['date'].min(),d['date'].max())