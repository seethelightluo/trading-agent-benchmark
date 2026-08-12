import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-08-20')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut,'close'] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); p=pd.DataFrame({s:x.reindex(idx) for s,x in P.items()}).ffill(); r=np.log(p).diff()
vol40=r.rolling(40,min_periods=25).std(); vol20=r.rolling(20,min_periods=15).std()
raw=(r.rolling(20,min_periods=15).sum()-r.rolling(60,min_periods=40).sum()/3)/(vol40+1e-8)-.35*r.rolling(5,min_periods=5).sum()/(vol20+1e-8)
f=raw.shift(1); f.to_csv('scripts/miner_2_20310821_trend_acceleration_quality_reval_signal.csv')
print('range',p.index.min().date(),p.index.max().date(),'assets',len(U),'rows',len(p))
for h in [1,5,10,20]:
 a=[];ns=[];ds=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(p.index[i])
 a=np.array(a);print('h',h,'obs',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 if h==20:
  d=pd.DatetimeIndex(ds)
  for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-28','2026','2028-12-31'),('2029-30','2029','2030-12-31'),('2031','2031','2031-12-31')]:
   q=a[(d>=pd.Timestamp(lo))&(d<=pd.Timestamp(hi))]; print('regime',name,'obs',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)) if len(q)>1 else 'insufficient')
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'valid_dates',int(f.notna().sum(axis=1).ge(8).sum()))
