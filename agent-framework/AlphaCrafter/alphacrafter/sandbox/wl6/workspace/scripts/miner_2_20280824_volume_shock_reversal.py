import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=4000)
   if x is not None and len(x): return x
  except Exception: pass
P={};V={}
for s in U:
 x=fetch(s)
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date);y=x.set_index('date').sort_index();P[s]=y.close.astype(float); V[s]=y.volume.astype(float) if 'volume' in y else pd.Series(index=y.index,dtype=float)
p=pd.DataFrame(P).sort_index(); volu=pd.DataFrame(V).reindex(p.index); r=p.pct_change(); sig=r.rolling(20,min_periods=12).std()
# Large abnormal one-day move with volume confirmation; reversal signal, lagged at decision
vs=volu/(volu.rolling(20,min_periods=12).median())-1
f=-(r/sig)*vs
frs={h:r.shift(-1).rolling(h).sum().shift(-(h-1)) for h in [1,5,10]}
for h,fr in frs.items():
 vals=[];ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>=3:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(dt)
 q=pd.Series(vals,index=dates).dropna(); print('horizon',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
   z=q[(q.index.astype(str)>=a)&(q.index.astype(str)<=b)];print('regime',a,'n',len(z),'IC',round(z.mean(),6) if len(z) else None)
# rank turnover
prev=None;ts=[]
for dt in f.index:
 z=f.loc[dt].dropna()
 if len(z)>=8:
  rk=z.rank(pct=True)
  if prev is not None:ts.append(abs(rk-prev.reindex(rk.index)).mean())
  prev=rk
print('coverage',round(np.mean([len(f.loc[d].dropna()) for d in f.index])/len(U),4),'turnover',round(np.mean(ts),5))
