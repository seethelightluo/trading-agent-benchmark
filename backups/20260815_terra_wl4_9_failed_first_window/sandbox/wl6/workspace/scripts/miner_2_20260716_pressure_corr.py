import pandas as pd,numpy as np,json
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; out=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); d=d.set_index('date'); z=(d.high-d.low).replace(0,np.nan); clv=2*(d.close-d.low)/z-1
 out.append(pd.DataFrame({'date':d.index,'s':s,'candidate':-clv,'rev3':-(d.close/d.close.shift(3)-1),'rev5':-(d.close/d.close.shift(5)-1),'mom':(d.close/d.close.shift(20)-1)/ (d.close.pct_change().rolling(60).std()*np.sqrt(20))}))
x=pd.concat(out).dropna(); print(x[['candidate','rev3','rev5','mom']].corr())
