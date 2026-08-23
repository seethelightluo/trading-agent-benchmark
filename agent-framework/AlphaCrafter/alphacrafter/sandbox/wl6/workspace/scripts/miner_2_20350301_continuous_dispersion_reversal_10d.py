import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d): return d
  except Exception: pass
P={}
for s in U:
 d=fetch(s)
 if d is not None:
  d=d.copy(); d['date']=pd.to_datetime(d.date); P[s]=d.set_index('date').close.astype(float)
p=pd.concat(P,axis=1).sort_index(); r=np.log(p).diff(); cs=r.std(axis=1,ddof=1)
pct=cs.rolling(252,min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
r10=p.pct_change(10); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
f=(-(r10/vol)*(0.5+pct.values[:,None])); f=pd.DataFrame(f,index=p.index,columns=p.columns).shift(1)
res=[]
for h in [5,10,20,40]:
 fw=p.shift(-h)/p-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): rows.append((dt,c,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); x=q.ic.to_numpy(); m=x.mean(); ir=m/x.std(ddof=1)*np.sqrt(len(x))
 print('h',h,'dates',len(x),'avg_names',q.n.mean(),'coverage',q.n.mean()/15,'IC',m,'ICIR',ir,'hit',np.mean(x>0))
 if h==10:
  for label,lo,hi in [('early','2020-01-01','2027-12-31'),('mid','2028-01-01','2031-12-31'),('recent','2032-01-01','2035-02-28')]:
   y=q[(q.date>=lo)&(q.date<=hi)].ic; print(label,len(y),y.mean())
