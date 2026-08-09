"""Miner 2: USDJPY shock-magnitude conditioned common-beta asymmetry validation."""
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2034-07-05')
def series(path,col='close'):
 d=pd.read_csv(path,parse_dates=['date']).set_index('date')[col]
 return d.loc[d.index<=END]
# completed-bar close returns; intersections avoid asynchronous accidental fills
px=pd.concat({a:series('../persistent/stock_data/'+a+'.csv') for a in ASSETS},axis=1).sort_index()
r=px.pct_change().replace([np.inf,-np.inf],np.nan)
fx=series('../persistent/index_data/USDJPY.csv').pct_change().reindex(r.index)
common=r.median(axis=1)
# One factor idea: 60-observation beta on large USDJPY shocks minus beta on ordinary FX days.
# The shock cutoff is trailing 60-day 70th percentile (strictly lagged).
out=pd.DataFrame(index=r.index,columns=ASSETS,dtype=float)
for t in range(61,len(r)):
 ix=r.index[t-60:t] # excludes decision day's return
 f=fx.iloc[t-60:t]
 cut=f.abs().quantile(.70)
 shock=f.abs()>=cut; calm=~shock
 for a in ASSETS:
  y=r[a].iloc[t-60:t]; x=common.iloc[t-60:t]
  ok=y.notna()&x.notna()&f.notna()
  def beta(mask):
   z=ok&mask
   return np.cov(y[z],x[z],ddof=1)[0,1]/np.var(x[z],ddof=1) if z.sum()>=10 and np.var(x[z])>0 else np.nan
  out.loc[r.index[t],a]=beta(shock)-beta(calm)

def metrics(h):
  ics=[]; dates=[]; ns=[]
  fwd=px.shift(-h)/px-1
  for dt in out.index:
   z=pd.concat([out.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
    if np.isfinite(v): ics.append(v);dates.append(dt);ns.append(len(z))
  x=np.array(ics); return dict(IC=float(x.mean()),ICIR=float(x.mean()/x.std(ddof=1)) if x.std(ddof=1)>0 else np.nan,hit=float((x>0).mean()),dates=len(x),mean_n=float(np.mean(ns)),raw=x,ds=pd.DatetimeIndex(dates))
print('FACTOR usd-jpy-shock-magnitude-conditioned-common-beta-asymmetry-60obs')
print('endpoint',END.date(),'factor_cells',int(out.notna().sum().sum()),'of',out.size,'coverage',round(out.notna().mean().mean(),6))
allm={}
for h in [1,5,10,20]:
 m=metrics(h);allm[h]=m; print('h',h,{k:round(v,6) if isinstance(v,float) else v for k,v in m.items() if k not in ('raw','ds')})
m=allm[20]
for label,lo,hi in [('2026-2029','2026-01-01','2029-12-31'),('2030-2032','2030-01-01','2032-12-31'),('2033-end','2033-01-01','2034-07-05')]:
 z=m['raw'][(m['ds']>=lo)&(m['ds']<=hi)]
 print('regime',label,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6))
# daily rank turnover
rank=out.rank(axis=1,pct=True)
print('turnover',round((rank-rank.shift()).abs().stack().mean(),6),'comparisons',int((rank-rank.shift()).stack().shape[0]),'median_iqr',round(out.quantile(.75,axis=1).sub(out.quantile(.25,axis=1)).median(),6))
# Persist signal matrix solely as audit artifact outside factor library for downstream exact orthogonality check
out.to_csv('scripts/miner_2_20340706_usdjpy_beta_signal.csv')
