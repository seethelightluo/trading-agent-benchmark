"""Validate one factor: peer-relative directional consistency (60 sessions)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def series(a,field='close'):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date)
 return pd.to_numeric(d.sort_values('date').set_index('date')[field],errors='coerce')
P=pd.DataFrame({a:series(a) for a in A}).sort_index(); r=P.pct_change(); rel=r.sub(r.median(1),axis=0)
# Persistently positive (negative) peer-relative daily direction is a simple, path-consistency trend signal.
# Magnitudes are deliberately discarded to prevent isolated outlier sessions from dominating.
F=rel.apply(np.sign).rolling(60,min_periods=40).mean().shift(1)
F=F.sub(F.median(1),axis=0); cut=P.index.max()
def stat(x,h,span=None):
 if span:x=x.loc[span[0]:span[1]]
 y=P.shift(-h).div(P).sub(1).reindex(x.index); z=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z); return dict(dates=len(z),ic=round(z.mean(),6),icir=round(z.mean()/z.std(ddof=1),6),hit=round((z>0).mean(),4),mean_n=round(np.mean(ns),2),min_n=int(min(ns)))
print('FACTOR peer_relative_directional_consistency_60 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20): print('H',h,stat(F,h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01','2028-12-31')),('2029_current',('2029-01-01',str(cut.date()))),('recent180',(str(cut-pd.Timedelta(days=180)),str(cut.date())))]: print('REGIME10',n,stat(F,10,s))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(1).mean()),6))
