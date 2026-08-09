"""One candidate: continuous broad-stress weighted peer resilience reversal (60).
For every day, weight an asset's peer-relative return by the continuously measured
magnitude of the cross-asset median risk-off move, with a modest extra multiplier
when broad stress has persisted over the preceding five sessions.  The negative
sign tests whether assets that were unusually resilient during broad stress then
mean-revert relative to peers. Unlike an event gate, every mature date has a
signal and weights vary smoothly with the macro state.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); cut=P.index.max()
# Broad move uses the asset median, so no single asset sets the state.
m=R.median(axis=1); msd=m.rolling(60,min_periods=40).std().replace(0,np.nan)
# Continuous, bounded downside intensity: zero in non-risk-off days, 1 at one sd.
stress=(-m/msd).clip(lower=0,upper=3)
# Persistent stress only reweights; it does not make cells unavailable.
persist=(stress>0.35).rolling(5,min_periods=1).sum().shift(1).fillna(0)
w=stress*(1+0.25*persist)
rel=R.sub(R.median(axis=1),axis=0)
# Min observations prevent a very small number of weighted days dominating.
den=w.rolling(60,min_periods=42).sum().replace(0,np.nan)
F=(-rel.mul(w,axis=0).rolling(60,min_periods=42).sum().div(den,axis=0))
F=F.sub(F.median(axis=1),axis=0).shift(1)
def metric(h,lo=None,hi=None,sign=1):
 x=(sign*F).loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); vals=[];ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z); ns.append(len(q))
 if not vals:return {'dates':0}
 vals=np.array(vals); return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR continuous_broad_stress_weighted_peer_resilience_reversal_60 cutoff',cut.date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6),'mean_stress',round(float(stress.mean()),5))
for s,n in [(1,'reversal'),(-1,'resilience_followthrough')]:
 print('ORIENTATION',n)
 for h in (1,5,10,20):print('H',h,metric(h,sign=s))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]:print('REGIME10',n,metric(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
print('LIBRARY_CORRELATION deferred unless aggregate and recent directional evidence pass.')
