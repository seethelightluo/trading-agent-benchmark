"""One idea: stealth peer-relative trend: mean peer-relative return earned on its own low-volume days."""
import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date).dt.normalize();d=d.sort_values('date').set_index('date')
 C[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
C=pd.DataFrame(C).sort_index();V=pd.DataFrame(V).reindex(C.index); R=C.pct_change(); peer=R.sub(R.median(axis=1),axis=0)
# own volume ratio's lower-than-its-60d median event; mean peer-relative returns, lagged.
vr=np.log(V/V.rolling(20,min_periods=15).mean()); ev=vr.lt(vr.rolling(60,min_periods=40).median())
F=peer.where(ev).rolling(60,min_periods=12).mean().shift(1); F=F.sub(F.median(axis=1),axis=0)
print('cutoff',C.index.max().date(),'calendar dates',len(C),'factor cells',int(F.notna().sum().sum()),'coverage',F.notna().mean().mean())
print('event rate',ev.mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().stack().mean(),'dispersion',F.std(axis=1).mean())
for h in [1,5,10,20]:
 y=C.shift(-h)/C-1; ics=[]; ns=[]; ds=[]
 for t in F.index:
  x=F.loc[t];z=y.loc[t];m=x.notna()&z.notna()
  if m.sum()>=8: ics.append(x[m].corr(z[m],method='spearman'));ns.append(m.sum());ds.append(t)
 q=np.array(ics); print(h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'breadth',np.mean(ns),'range',min(ds).date(),max(ds).date())
 for label,mask in [('2023-2026',pd.DatetimeIndex(ds)<pd.Timestamp('2027-01-01')),('2027+',pd.DatetimeIndex(ds)>=pd.Timestamp('2027-01-01')),('recent180',pd.DatetimeIndex(ds)>=C.index.max()-pd.Timedelta(days=180))]:
  a=q[mask]
  print(' ',label,len(a),a.mean() if len(a) else None,a.mean()/a.std(ddof=1) if len(a)>1 else None,(a>0).mean() if len(a) else None)
