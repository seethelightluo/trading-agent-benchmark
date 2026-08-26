import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-08-28')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=end].set_index('date'); P[s]=d.close
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
# Recovery quality: lagged 20-session return, penalized by downside deviation, and gated by positive lagged 60-session trend.
down=r.where(r<0).rolling(40,min_periods=20).std(); mom=px.shift(5)/px.shift(25)-1; trend=px.shift(5)/px.shift(65)-1
sig=(mom/(down*np.sqrt(252)+0.01)*np.where(trend>0,1,0.25)).shift(1)
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; z=[];ns=[]; ds=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
   if np.isfinite(q):z.append(q);ns.append(len(a));ds.append(dt)
 z=np.array(z); print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
 if h==10:
  rank=sig.rank(axis=1,pct=True); t=rank.diff().abs().mean(axis=1).dropna().mean(); print('TURN',round(t,6),'COVERAGE',round(sig.notna().sum().sum()/sig.size,4),'assets',len(U))
  for lab,a,b in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-28','2026','2028-12-31')]:
   q=z[[a<=str(d.date())<=b for d in ds]];print('REG',lab,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
sig.index.name='date';sig.to_csv('scripts/miner_2_20280828_recovery_quality_signal.csv')
print('range',px.index.min().date(),px.index.max().date())
