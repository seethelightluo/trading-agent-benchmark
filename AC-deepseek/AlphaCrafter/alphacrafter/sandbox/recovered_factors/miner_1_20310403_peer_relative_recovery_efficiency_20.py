"""One candidate: peer-relative recovery efficiency (20 sessions).
A high 20-session return achieved with a small cumulative absolute-return path
may reflect durable, rather than jump-driven, cross-asset leadership.  The
signal is each asset's return/path-efficiency minus the contemporaneous median
peer efficiency, delayed one completed session.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index();R=P.pct_change(); cutoff=P.index.max()
# Strictly t information, then execution-safe one session signal delay.
r20=P.pct_change(20); path=R.abs().rolling(20,min_periods=16).sum()
eff=r20/path.replace(0,np.nan)
F=eff.sub(eff.median(axis=1),axis=0).shift(1)
def met(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); vals=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   c=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(c):vals.append(c);ns.append(len(q))
 if not vals:return {'dates':0}
 v=np.array(vals); return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR peer_relative_recovery_efficiency_20 cutoff',cutoff.date(),'assets',len(A),'dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20):print('H',h,met(h))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cutoff-pd.Timedelta(days=180)),None)]:print('REGIME10',n,met(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTION_SD',round(float(F.std(axis=1).mean()),6))
print('LIBRARY_CORRELATION pending: only calculated after numerical and temporal-stability gates pass.')
