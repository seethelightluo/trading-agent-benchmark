import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); vs=v.set_index('date').close.reindex(p.index).ffill()
vr=vs.pct_change(5); stress=(vr/vr.rolling(60).std()).clip(-3,3).fillna(0); sig=-p.pct_change(5).mul(1+0.5*stress.clip(lower=0),axis=0); rows=[]
for i,t in enumerate(p.index):
 if i<70 or i+20>=len(p): continue
 for h in [1,5,10,20]:
  q=pd.concat([sig.iloc[i],p.shift(-h).iloc[i]/p.iloc[i]-1],axis=1).dropna(); q.columns=['s','f']
  if len(q)>=8 and q.s.nunique()>1 and q.f.nunique()>1: rows.append((t,h,len(q),q.s.rank().corr(q.f.rank())))
A=pd.DataFrame(rows,columns=['date','h','n','ic']); print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'rows',len(A))
for h in [1,5,10,20]:
 q=A[A.h==h]; x=q.ic; print('H',h,'dates',len(q),'mean_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for lab,cond in [('recent252',q.date>=q.date.max()-pd.Timedelta(days=370)),('online',q.date>=pd.Timestamp('2026-07-16')),('ytd',q.date>=pd.Timestamp('2028-01-01'))]:
  y=q[cond].ic; print(lab,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6) if len(y)>1 else np.nan)
