import pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for s in U:
 fn='../persistent/index_data/'+s+'.csv';fn=fn if os.path.exists(fn) else '../persistent/stock_data/'+s+'.csv'
 x=pd.read_csv(fn);print(s,len(x),x.date.iloc[0],x.date.iloc[-1])
