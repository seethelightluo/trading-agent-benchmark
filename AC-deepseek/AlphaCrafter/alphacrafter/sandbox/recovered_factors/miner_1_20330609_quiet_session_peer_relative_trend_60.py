"""One idea: quiet-session peer-relative trend: relative return earned on own unusually quiet days."""
import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date).dt.normalize();C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
C=pd.DataFrame(C).sort_index();R=C.pct_change(); peer=R.sub(R.median(axis=1),axis=0)
ev=R.abs().lt(R.abs().rolling(60,min_periods=40).median())
F=peer.where(ev).rolling(60,min_periods=12).mean().shift(1); F=F.sub(F.median(axis=1),axis=0)
print('cutoff',C.index.max().date(),'calendar dates',len(C),'factor cells',int(F.notna().sum().sum()),'coverage',F.notna().mean().mean())
print('event rate',ev.mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().stack().mean(),'dispersion',F.std(axis=1).mean())
for h in [1,5,10,20]:
 y=C.shift(-h)/C-1; out=[]
 for t in F.index:
  m=F.loc[t].notna()&y.loc[t].notna()
  if m.sum()>=8: out.append((t,F.loc[t,m].corr(y.loc[t,m],method='spearman'),m.sum()))
 ds=pd.DatetimeIndex([x[0] for x in out]);q=np.array([x[1] for x in out]);ns=[x[2] for x in out]
 print(h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'breadth',np.mean(ns),'range',ds.min().date(),ds.max().date())
 for label,mask in [('2023-2026',ds<pd.Timestamp('2027-01-01')),('2027+',ds>=pd.Timestamp('2027-01-01')),('recent180',ds>=C.index.max()-pd.Timedelta(days=180))]:
  a=q[mask];print(' ',label,len(a),a.mean() if len(a) else None,a.mean()/a.std(ddof=1) if len(a)>1 else None,(a>0).mean() if len(a) else None)
