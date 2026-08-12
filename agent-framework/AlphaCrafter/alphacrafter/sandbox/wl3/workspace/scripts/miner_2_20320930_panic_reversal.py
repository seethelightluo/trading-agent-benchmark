import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; frames={}
for s in U:
 d=get_stock_daily_data(s,days=6000); frames[s]=d.set_index(pd.to_datetime(d.date)).close
p=pd.DataFrame(frames).sort_index().ffill(); r=np.log(p).diff(); v=get_index_daily_data('VIX',days=6000); v=v.set_index(pd.to_datetime(v.date)).close.reindex(p.index).ffill()
pct=v.rolling(252,min_periods=126).apply(lambda x: np.mean(x<=x[-1]))
sig=(-r.rolling(5).sum()).mul(.5+pct,axis=0).rolling(3).mean().shift(1); fwd=p.shift(-10)/p-1
ics=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(q): ics.append(q);ns.append(len(z))
ics=np.array(ics); tr=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print('dates',len(ics),'assets',len(U),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0),'turnover',tr)
for lab,m in [('120',np.arange(len(ics))>=len(ics)-120),('252',np.arange(len(ics))>=len(ics)-252),('early',np.arange(len(ics))<len(ics)//2),('late',np.arange(len(ics))>=len(ics)//2)]:
 a=ics[m]; print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1))
