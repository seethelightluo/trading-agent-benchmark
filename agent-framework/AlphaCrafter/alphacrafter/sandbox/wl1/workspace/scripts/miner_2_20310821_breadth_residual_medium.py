import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-08-20')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 P[s]=d.loc[:cut,'close']
idx=sorted(set().union(*[set(x.index) for x in P.values()]))
p=pd.DataFrame({s:x.reindex(idx) for s,x in P.items()}).ffill(); r=p.pct_change()
# Candidate: continuous breadth-conditioned residual medium momentum, lagged one day.
# Relative 30d return is demeaned by cross-asset median and scaled by 40d realized vol.
ret=r.rolling(30,min_periods=20).sum(); rel=ret.sub(ret.median(axis=1),axis=0)
vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
breadth=(r.gt(0).rolling(40,min_periods=25).mean().mean(axis=1)).clip(.15,.85)
# Smooth breadth multiplier rewards leadership in broad participation and dampens narrow rallies.
f=(rel/(vol+0.004)).mul((0.5+breadth),axis=0).shift(1)
f.to_csv('scripts/miner_2_20310821_breadth_residual_medium_signal.csv')
print('range',p.index.min().date(),p.index.max().date(),'cutoff',cut.date(),'assets',p.shape[1],'rows',len(p))
for h in [1,5,10,20]:
 vals=[]; ns=[]; dates=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));dates.append(p.index[i])
 a=np.asarray(vals); print('h',h,'obs',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 if h==20:
  ds=pd.DatetimeIndex(dates)
  for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-28','2026','2028-12-31'),('2029-30','2029','2030-12-31'),('2031','2031','2031-12-31')]:
   q=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))]
   print('regime',name,'obs',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)) if len(q)>1 else 'insufficient')
rk=f.rank(axis=1,pct=True); turn=rk.diff().abs().mean(axis=1).dropna(); print('turnover_proxy',round(turn.mean(),6),'coverage_all',round(f.notna().mean().mean(),6),'valid_dates',f.notna().sum(axis=1).ge(8).sum())
