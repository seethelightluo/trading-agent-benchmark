"""One candidate: dispersion-state-conditioned peer-relative reversal.
In periods of unusually high cross-asset return dispersion, recent relative
moves can overshoot; test a volatility-normalized five-session relative-return
reversal only in that observable state.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); cutoff=P.index.max()
# State uses only realized same-day cross-asset dispersion and is delayed together with signal.
disp=R.std(axis=1); threshold=disp.rolling(60,min_periods=40).quantile(.70)
state=(disp>threshold)&(disp>disp.shift(5))
vol=R.rolling(20,min_periods=15).std().replace(0,np.nan)
relative=P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0)
F=(-relative/vol).sub((-relative/vol).median(axis=1),axis=0).where(state,axis=0).shift(1)
def met(h,lo=None,hi=None):
 x=F.loc[lo:hi];y=(P.shift(-h)/P-1).reindex(x.index); vals=[];ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z):vals.append(z);ns.append(len(q))
 if not vals:return {'dates':0}
 v=np.array(vals);return {'dates':len(v),'ic':round(v.mean(),6),'icir':round(v.mean()/v.std(ddof=1),6),'hit':round((v>0).mean(),6),'breadth':round(np.mean(ns),3),'min_breadth':min(ns)}
print('FACTOR dispersion_state_peer_relative_reversal_5_20 cutoff',cutoff.date(),'assets',len(A),'history_dates',len(P))
print('STATE_DATES',int(state.sum()),'STATE_RATE',round(float(state.mean()),6),'CELLS',int(F.notna().sum().sum()),'/',F.size,'COVERAGE',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20):print('H',h,met(h))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cutoff-pd.Timedelta(days=180)),None)]:print('REGIME10',n,met(10,lo,hi))
print('TURNOVER_ACTIVE',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD_ACTIVE',round(float(F.std(axis=1).mean()),6))
print('LIBRARY_CORRELATION pending unless pooled gates and directional regime stability pass.')
