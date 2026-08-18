import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); m=r.mean(axis=1)
# Relative residual return: remove common cross-asset movement using rolling beta.
def resid(h):
 cov=r.rolling(h,min_periods=max(10,h//2)).cov(m)
 var=m.rolling(h,min_periods=max(10,h//2)).var()
 beta=cov.div(var,axis=0)
 return r.rolling(h,min_periods=max(10,h//2)).sum()-beta.mul(m.rolling(h,min_periods=max(10,h//2)).sum(),axis=0)
# Candidate: residual medium reversal, risk-adjusted and conditioned on downside persistence.
res=resid(40)
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
down=r.clip(upper=0).rolling(30,min_periods=20).std()*np.sqrt(30)
# assets with unusually persistent negative days receive stronger recovery signal
neg=r.lt(0).rolling(30,min_periods=20).mean()
neg_rank=neg.rank(axis=1,pct=True)
base=-res/(vol+down+1e-6)
variants={'residual_reversal':base,'downside_conditioned':base*(0.5+neg_rank),'inverse_downside_conditioned':base*(1.5-neg_rank)}
fr=px.pct_change(10).shift(-10)
for name,f0 in variants.items():
 f=f0.shift(1); ics=[]; ns=[]; dates=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if np.isfinite(c): ics.append(c);ns.append(len(a));dates.append(dt)
 z=np.array(ics); rank=f.rank(axis=1,pct=True)
 print(name,'dates',len(z),'avgN',round(np.mean(ns),2),'IC %.8f ICIR %.8f hit %.5f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)),'coverage %.5f turnover %.5f'%(f.notna().sum(axis=1).mean()/len(U),rank.diff().abs().mean(axis=1).dropna().mean()))
 for start,end in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
  q=np.array([v for d,v in zip(dates,ics) if start<=str(d.year)<=end]); print(' REG',start,end,len(q),round(q.mean(),7),round(q.mean()/q.std(ddof=1),7))
 if name=='downside_conditioned':
  f.loc[dates].to_csv('scripts/miner_3_20340303_downside_conditioned_residual_signal.csv',index_label='date')
