import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-11-14')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']); px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill()
# Positive five-day rebound following a 60d drawdown, activated only in elevated VIX regimes.
dd=p/p.rolling(60,min_periods=40).max()-1; rebound=r.rolling(5).sum()
base=(rebound.clip(lower=0)*(-dd).clip(lower=0)).where(dd<0,0.)
gate=(v>v.rolling(120,min_periods=60).quantile(.80)).astype(float)
s=base.mul(gate,axis=0).rank(axis=1,pct=True)
print('rows',len(p),'assets',len(U),'cut',cut.date())
for h in [5,10,20]:
 xs=[];ns=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: xs.append(q.f.corr(q.y,method='spearman'));ns.append(len(q));ds.append(p.index[i])
 x=pd.Series(xs,index=ds); print('TEST',h,'dates',len(x),'IC',round(x.mean(),8),'ICIR',round(x.mean()/x.std(ddof=1),8),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4))
# write causal artifact for the admission horizon (10d)
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20291115_macro_stress_recovery_signal.csv',index=False)
print('artifact_rows',len(out),'turnover',s.diff().abs().mean().mean(),'coverage',s.notna().mean().mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-11-14')]:
 xs=[]
 for i in range(len(p)-10):
  if pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b):
   q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1:xs.append(q.f.corr(q.y,method='spearman'))
 q=pd.Series(xs);print('REG10',a,b,'dates',len(q),'IC',round(q.mean(),8),'ICIR',round(q.mean()/q.std(ddof=1),8) if len(q)>1 else None)
print('max_abs_library_correlation',None)
