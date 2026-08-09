import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2027-02-24'
F={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); F[s]=d[d.date<=end].set_index('date').close
P=pd.concat(F,axis=1).sort_index(); R=P.pct_change()
# Short-horizon return scaled by trailing realized risk, with one-day information lag.
for w in [5,10,20]:
 sig=(P.pct_change(w)/R.rolling(20).std()).shift(1); fwd=R.shift(-1)
 ics=[]; ds=[]; ns=[]; ranks=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   ics.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ds.append(dt); ns.append(len(q)); ranks.append(sig.loc[dt].rank())
 ic=pd.Series(ics,index=pd.to_datetime(ds)); sd=ic.std(ddof=1)
 print('W',w,'dates',len(ic),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(ic.mean(),8),'ICIR',round(ic.mean()/sd*np.sqrt(252),8),'hit',round((ic>0).mean(),4),'turn',round(np.nanmean([(ranks[j]-ranks[j-1]).abs().sum()/225 for j in range(1,len(ranks))]),4))
 for label,a,b in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('recent','2026-07-16',end)]:
  q=ic[(ic.index>=a)&(ic.index<=b)]; print(' ',label,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),4) if len(q)>2 else np.nan)
