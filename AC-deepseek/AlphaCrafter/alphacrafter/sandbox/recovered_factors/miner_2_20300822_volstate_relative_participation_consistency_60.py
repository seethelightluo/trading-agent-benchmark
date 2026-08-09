"""Miner 2: volatility-state-conditioned relative participation consistency.
For each asset, measure its signed directional agreement with the daily median-peer
return over 60 days, weighting observations by a lagged broad-volatility state.
The factor is a path-participation reliability feature rather than return level,
drawdown, tail event, or raw beta.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); peer=R.median(axis=1); cut=P.index.max()
# State weight uses volatility information fully known before the scored date.
bvol=peer.rolling(15,min_periods=12).std(); state=(bvol/bvol.rolling(60,min_periods=45).median()).clip(.5,2.0).shift(1)
# Continuous normalized signed co-movement avoids a threshold and handles scale differences.
asset_z=R.div(R.rolling(60,min_periods=45).std().replace(0,np.nan))
peer_z=peer.div(peer.rolling(60,min_periods=45).std().replace(0,np.nan))
agreement=np.tanh(asset_z).mul(np.tanh(peer_z),axis=0)
F=agreement.mul(state,axis=0).rolling(60,min_periods=45).mean()
F=F.sub(F.median(axis=1),axis=0).shift(1)
def metric(h,lo=None,hi=None,sign=1):
 x=(sign*F).loc[lo:hi]; y=(P.shift(-h).div(P)-1).reindex(x.index); vals=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);ns.append(len(q))
 if not vals:return {'dates':0}
 vals=np.array(vals);return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR volatility_state_conditioned_relative_participation_consistency_60 cutoff',cut.date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for s,n in [(1,'high_participation'),(-1,'low_participation')]:
 print('ORIENTATION',n)
 for h in (1,5,10,20):print('H',h,metric(h,sign=s))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]: print('REGIME10',n,metric(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
print('Library correlation deferred unless predictive admission gates pass.')
