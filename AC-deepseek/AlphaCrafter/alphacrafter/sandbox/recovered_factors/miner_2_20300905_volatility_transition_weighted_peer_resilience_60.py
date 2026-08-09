"""Miner 2 single-idea validation: broad-volatility-transition weighted peer resilience, 60 sessions."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def close(a):
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame({a:close(a) for a in A}).sort_index(); R=P.pct_change(); M=R.median(axis=1); REL=R.sub(M,axis=0)
# State-transition weight: change in lagged 15d broad volatility relative to its 60d median.
# Positive score means peer-relative resilience accumulated as aggregate risk transitions upward.
bv=M.rolling(15,min_periods=10).std(); state=bv/bv.rolling(60,min_periods=35).median(); transition=state.diff().clip(lower=0)
F=REL.mul(transition.shift(1),axis=0).rolling(60,min_periods=18).sum()
F=F.sub(F.median(axis=1),axis=0).shift(1)
def met(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=P.shift(-h).div(P).sub(1).reindex(x.index); z=[]; nn=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   c=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(c):z.append(c);nn.append(len(q))
 z=np.asarray(z)
 return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),4),'mean_n':round(np.mean(nn),2),'min_n':min(nn)} if len(z) else {'dates':0}
print('FACTOR volatility_transition_weighted_peer_resilience_60 cutoff',P.index.max().date(),'assets',len(A))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(F.notna().stack().mean(),6),'transition_days',int(transition.notna().sum()),'positive_transition_days',int((transition>0).sum()))
for h in (1,5,10,20):print('H',h,met(h))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(P.index.max()-pd.Timedelta(days=180)),None)]:print('REGIME10',n,met(10,lo,hi))
print('TURNOVER',round(F.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CROSS_SECTIONAL_SD',round(F.std(axis=1).mean(),6))
print('LIBRARY_CORRELATION_EVIDENCE','Deferred unless predictive gates and recent robustness pass; full exact audit required before admission.')
