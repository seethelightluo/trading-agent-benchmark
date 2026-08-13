import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000).copy(); d.date=pd.to_datetime(d.date)
 return d.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:load(s) for s in U}).sort_index(); R=P.pct_change()
# Medium-horizon residual reversal, normalized by recent idiosyncratic volatility.
res5=R.rolling(5,min_periods=5).sum().sub(R.rolling(5,min_periods=5).sum().median(axis=1),axis=0)
vol=R.rolling(20,min_periods=15).std()
disp=R.rolling(5,min_periods=5).std().mean(axis=1)
q=disp.shift(1).rolling(252,min_periods=100).quantile(.60)
gate=disp.shift(1)>q
F=(-res5/vol).shift(1).where(gate,np.nan); F.index.name='date'
F.to_csv('scripts/miner_2_20330203_dispersion60_residual5_signal.csv')
print('total_dates',len(P),'active_dates',int(gate.sum()),'assets',len(U),'mean_coverage_active',np.mean(F.notna().mean(axis=1).where(gate)))
for h in [1,3,5,10]:
 vals=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z))
 a=np.asarray(vals); a=a[np.isfinite(a)]
 print('horizon',h,'dates',len(a),'avg_instruments',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
rank=F.rank(axis=1,pct=True); print('rank_turnover_active',np.nanmean(((rank-rank.shift(1)).abs().mean(axis=1)).where(gate)))
for lo,hi in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-12-31')]:
 x=[]
 for i in np.where((F.index>=lo)&(F.index<=hi))[0]:
  if i+1>=len(P):continue
  z=pd.concat([F.iloc[i],P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:x.append(z.iloc[:,0].corr(z.iloc[:,1]))
 x=np.asarray(x);x=x[np.isfinite(x)]
 print('regime',lo,hi,'dates',len(x),'IC',np.mean(x) if len(x) else np.nan,'ICIR',np.mean(x)/np.std(x,ddof=1) if len(x)>1 else np.nan)
