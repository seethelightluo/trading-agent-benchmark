import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None: px[s]=d.set_index(pd.to_datetime(d.date)).close
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); cs=r.mean(axis=1); disp=r.std(axis=1).rolling(20).mean().shift(1); threshold=disp.rolling(252).median().shift(1)
raw=r.rolling(5).sum().shift(1); f=-raw.div(r.rolling(20).std().shift(1)); f=f.where(disp.gt(threshold),0.0)
fr=p.shift(-10)/p-1; vals=[]; dates=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));dates.append(dt);ns.append(len(z))
a=pd.Series(vals,index=dates).dropna(); print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean())
for h in [1,5,20]:
 fr=p.shift(-h)/p-1; v=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(v).dropna(); print('decay',h,q.mean(),q.mean()/q.std(ddof=1))
