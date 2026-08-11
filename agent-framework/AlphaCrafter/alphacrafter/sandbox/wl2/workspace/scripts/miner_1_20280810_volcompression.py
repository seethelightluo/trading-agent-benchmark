import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 try:d=get_stock_daily_data(s,days=4000)
 except:continue
 if d is not None and len(d):
  x=d[['date','close']].dropna();x.date=pd.to_datetime(x.date);px[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# volatility compression: prior 10d realized vol relative to prior 40d vol. Lower ratio scores higher.
v=r.rolling(10,min_periods=8).std()/r.rolling(40,min_periods=30).std(); f=-v
for h in [1,5,10,20]:
 out=[]
 for i in range(len(p)-h-1):
  q=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:out.append(q.iloc[:,0].corr(q.iloc[:,1]))
 a=pd.Series(out);print('h',h,'dates',len(a),'avgN',len(px),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('coverage',f.notna().sum(axis=1).mean()/15,'period',p.index.min(),p.index.max())
