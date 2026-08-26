import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=None
 for f in (get_index_daily_data,get_stock_daily_data):
  try: d=f(s,4200)
  except Exception: d=None
  if d is not None: break
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in P.items()}); r=cl.pct_change(); v=r.rolling(40,min_periods=20).std()
base=(.4*(cl/cl.shift(20)-1)/(v*np.sqrt(20))+.6*(cl/cl.shift(60)-1)/(v*np.sqrt(60)))
b=(r.rolling(20).sum()>0).mean(axis=1).shift(1)
base=base.mul(np.sign(2*b-1)*np.sqrt(np.abs(2*b-1)),axis=0).shift(1)
m=r.mean(axis=1)
beta=r.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0).shift(1)
sig=base.copy()
for dt in base.index:
 a=pd.concat([base.loc[dt],beta.loc[dt]],axis=1).dropna()
 if len(a)>=8 and a.iloc[:,1].std()>0:
  z=a.iloc[:,1].values; y=a.iloc[:,0].values
  slope=np.cov(z,y,ddof=1)[0,1]/np.var(z,ddof=1)
  sig.loc[dt,a.index]=y-slope*(z-z.mean())
# five-session smoothing uses only signals already formed from lagged inputs
sig=sig.rolling(5,min_periods=3).mean()
vals=[]; ns=[]; dates=[]
for dt in sig.index:
 y=cl.shift(-20).loc[dt]/cl.shift(-1).loc[dt]-1
 z=pd.concat([sig.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): vals.append(q);ns.append(len(z));dates.append(dt)
q=pd.Series(vals,index=dates)
print('assets',len(cl.columns),'rows',len(cl),'valid_dates',len(q),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().mean().mean(),4))
print('IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'recent756_ICIR',round(q.tail(756).mean()/q.tail(756).std(ddof=1),6))
for a,b0 in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-12-31'),('2035','2035-09-30')]:
 x=q.loc[a:b0]
 if len(x): print('regime',a,'n',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
sig.to_csv('scripts/miner_2_20351029_beta_residual_trend_smooth5_signal.csv')
