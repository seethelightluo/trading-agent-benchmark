"""One candidate: peer-market beta residual trend (20d return; 60d beta).
For each asset, isolate its 20-session move from the move predicted by its
rolling beta to an equal-weighted peer market. This tests whether idiosyncratic
cross-asset leadership persists rather than simply loading on common risk.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); cutoff=P.index.max()
# Each asset's peer market excludes itself, avoiding direct mechanical self-loading.
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 peer=R.drop(columns=a).median(axis=1)
 beta=R[a].rolling(60,min_periods=45).cov(peer)/peer.rolling(60,min_periods=45).var().replace(0,np.nan)
 residual=P[a].pct_change(20)-beta*peer.rolling(20,min_periods=16).sum()
 F[a]=residual
F=F.sub(F.median(axis=1),axis=0).shift(1)
def met(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); vals=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z);ns.append(len(q))
 if not vals:return {'dates':0}
 v=np.array(vals);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR peer_market_beta_residual_trend_20_60 cutoff',cutoff.date(),'assets',len(A),'dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20):print('H',h,met(h))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cutoff-pd.Timedelta(days=180)),None)]: print('REGIME10',n,met(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTION_SD',round(float(F.std(axis=1).mean()),6))
print('LIBRARY_CORRELATION pending: only calculated after numerical and temporal-stability gates pass.')
