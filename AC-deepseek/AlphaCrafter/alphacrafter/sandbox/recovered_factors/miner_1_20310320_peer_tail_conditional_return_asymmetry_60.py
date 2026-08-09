"""One candidate: peer-tail conditional return asymmetry (60 sessions).
Score assets by their idiosyncratic mean return on broad-peer downside days minus
an equal-sized penalty for weak performance on broad-peer upside days.  This is a
conditional co-movement/resilience characteristic, not a price-trend level.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def series(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date']).dt.normalize()
 return pd.Series(pd.to_numeric(d['close'],errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:series(a) for a in A}).sort_index(); R=P.pct_change()
# Leave-one-out peer return prevents each asset from mechanically entering its state.
peer=pd.DataFrame({a:R.drop(columns=a).median(axis=1) for a in A})
# Each asset's conditional residual return: behavior on its peers' worst/best 30% days.
def calc(a):
 qlo=peer[a].rolling(60,min_periods=45).quantile(.30); qhi=peer[a].rolling(60,min_periods=45).quantile(.70)
 low=R[a].where(peer[a].lt(qlo)).rolling(60,min_periods=45).mean()
 high=R[a].where(peer[a].gt(qhi)).rolling(60,min_periods=45).mean()
 # Favor less-negative / positive downside-day return and penalize negative upside-day return.
 return low-high.clip(upper=0)
F=pd.DataFrame({a:calc(a) for a in A}).shift(1)
F=F.sub(F.median(axis=1),axis=0)
def met(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); out=[]; ns=[]
 for t in x.index:
  z=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>2:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v):out.append(v);ns.append(len(z))
 if not out:return {'dates':0}
 out=np.array(out); return {'dates':len(out),'ic':round(out.mean(),6),'icir':round(out.mean()/out.std(ddof=1),6),'hit':round((out>0).mean(),4),'mean_n':round(np.mean(ns),2),'min_n':min(ns)}
cut=P.index.max();print('FACTOR peer_tail_conditional_return_asymmetry_60 cutoff',cut.date(),'assets',len(A),'price_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(F.notna().stack().mean(),6))
for h in (1,5,10,20):print('H',h,met(h))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]: print('REGIME10',n,met(10,lo,hi))
print('TURNOVER',round(F.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'XS_SD',round(F.std(axis=1).mean(),6))
print('LIBRARY_CORRELATION requires complete executable signal reconstruction only if gates pass.')
