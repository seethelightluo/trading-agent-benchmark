import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 D[s]=p
px=pd.concat(D,axis=1).sort_index()
# acceleration: recent 5d return versus prior 15d average daily return, normalized by 20d vol
r5=px/px.shift(5)-1
r20=px/px.shift(20)-1
vol=px.pct_change().rolling(20).std()
f=(r5-r20/4)/vol.replace(0,np.nan)
rows=[]
for h in [5,10,20]:
  ics=[]; dates=[]; n=[]
  for dt in f.index:
    fut=px.shift(-h).loc[dt]/px.loc[dt]-1
    x=f.loc[dt]; z=pd.concat([x,fut],axis=1).dropna()
    if len(z)>=8:
      ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
      if np.isfinite(ic): ics.append(ic);dates.append(dt);n.append(len(z))
  a=np.array(ics); print(h,'dates',len(a),'avg_n',np.mean(n),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',np.mean([len(px.loc[d].dropna()) for d in dates])/15)
  if h==10:
   # turnover rank proxy
   q=f.rank(axis=1,pct=True); print('turnover',q.diff().abs().mean(axis=1).loc[dates].mean())
   for y in sorted(set(d.year for d in dates)):
    aa=a[[d.year==y for d in dates]]; print(y,round(aa.mean(),5),len(aa))
