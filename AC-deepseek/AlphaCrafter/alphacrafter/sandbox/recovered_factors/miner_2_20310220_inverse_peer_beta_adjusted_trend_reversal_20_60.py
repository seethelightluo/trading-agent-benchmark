"""Miner 2: one candidate — inverse peer-beta-adjusted 20-session trend residual.
For each asset, regress its daily returns on the equal-weighted return of the
other available assets over 60 completed sessions. The factor is the negative
of its beta-adjusted 20-session cumulative return, divided by own 20-session
realized volatility. It tests residual (rather than raw) medium-short reversal.
All values are lagged one session before forward-return testing.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

assets=get_account_dict()['watch_list']
def close(a):
    d=get_stock_daily_data(a,5000).copy()
    d['date']=pd.to_datetime(d['date']).dt.normalize()
    return pd.Series(pd.to_numeric(d['close'],errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:close(a) for a in assets}).sort_index()
R=P.pct_change()
# Leave-one-out peer series makes beta independent of the asset's own shock.
peer=pd.DataFrame({a:R.drop(columns=a).mean(axis=1) for a in assets})
beta=pd.DataFrame(index=P.index,columns=assets,dtype=float)
for a in assets:
    cov=R[a].rolling(60,min_periods=45).cov(peer[a])
    var=peer[a].rolling(60,min_periods=45).var()
    beta[a]=cov/var.replace(0,np.nan)
raw20=R.rolling(20,min_periods=15).sum()
peer20=peer.rolling(20,min_periods=15).sum()
vol20=R.rolling(20,min_periods=15).std().replace(0,np.nan)
F=-(raw20-beta*peer20)/vol20
F=F.sub(F.median(axis=1),axis=0).shift(1)
cutoff=P.index.max()
def metrics(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); vals=[]; breadth=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);breadth.append(len(q))
 if not vals:return {'dates':0}
 vals=np.array(vals)
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),4),'mean_n':round(float(np.mean(breadth)),2),'min_n':int(min(breadth))}
print('FACTOR inverse_peer_beta_adjusted_trend_reversal_20_60 cutoff',cutoff.date(),'assets',len(assets),'calendar_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for h in [1,5,10,20]:print('H',h,metrics(h))
print('REGIMES horizon10')
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cutoff-pd.Timedelta(days=180)),None)]:print(n,metrics(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTION_SD',round(float(F.std(axis=1).mean()),6))
