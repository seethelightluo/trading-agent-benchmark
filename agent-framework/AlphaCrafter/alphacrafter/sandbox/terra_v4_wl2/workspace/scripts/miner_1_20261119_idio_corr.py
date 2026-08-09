import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-11-18')
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}; p=pd.DataFrame(p).sort_index(); r=p.pct_change(); m=r.median(axis=1)
f=-(r.sub(m,axis=0)).rolling(2,min_periods=2).sum()
proxies={'peer':r.rolling(5).median().shift(0),'rev5':-r.rolling(5).sum(),'mom20':r.rolling(20).sum()/(r.rolling(20).std()*np.sqrt(20))}
# clv unavailable without OHLC; report max among direct return proxies
for k,v in proxies.items():
 z=pd.concat([f.stack(),v.stack()],axis=1).dropna(); print(k,len(z),z.corr().iloc[0,1])
