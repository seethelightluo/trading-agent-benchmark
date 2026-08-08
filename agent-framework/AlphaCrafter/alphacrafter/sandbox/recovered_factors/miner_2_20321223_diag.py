import pandas as pd
for a in ['SPX','BTC','US10Y']:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date'])
 print(a,len(d),d.date.min(),d.date.max(),d.close.notna().sum())
