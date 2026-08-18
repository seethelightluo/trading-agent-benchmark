import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date') for s in U}; p=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index();o=pd.DataFrame({s:d.open for s,d in D.items()}).reindex(p.index); r=p.pct_change()
g=o/p.shift(1)-1
for w in [3,5,10]:
 f=-g.rolling(w,min_periods=2).mean();ic=[];ns=[];ds=[]
 for i in range(len(p)-1):
  q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8:ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q));ds.append(p.index[i])
 x=np.array(ic); print(w,len(x),round(np.mean(ns),2),round(x.mean(),5),round(x.mean()/x.std(ddof=1),5),round((x>0).mean(),4),round(np.mean(ns)/15,3))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=x[(pd.DatetimeIndex(ds).year>=lo)&(pd.DatetimeIndex(ds).year<=hi)];print(lo,round(z.mean(),5),len(z))
