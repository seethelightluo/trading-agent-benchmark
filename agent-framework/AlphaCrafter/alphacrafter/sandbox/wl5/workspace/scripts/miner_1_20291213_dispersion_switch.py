import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2029-12-12');px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna(how='all');r=p.pct_change();mom=p.pct_change(20)
# Causal dispersion regime: cross-asset average 20d realized vol relative to trailing median.
disp=r.rolling(20,min_periods=15).std().mean(axis=1); med=disp.rolling(252,min_periods=100).median(); high=(disp>med).astype(float)
# trend in calm regimes, reversal in high-dispersion regimes
s=mom.mul(1-2*high,axis=0)
for h in [5,10,20]:
 xs=[];ns=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:xs.append(q.f.corr(q.y,method='spearman'));ns.append(len(q))
 x=pd.Series(xs);print('TEST',h,'dates',len(x),'meanN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4))
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20291213_dispersion_switch_signal.csv',index=False)
print('rows',len(p),'assets',len(U),'cut',cut.date(),'turnover',round(s.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'coverage',round(s.notna().mean().mean(),4))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-12-12')]:
 for h in [10]:
  xs=[]
  for i in range(len(p)-h):
   if pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b):
    q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
    if len(q)>=8 and q.f.nunique()>1:xs.append(q.f.corr(q.y,method='spearman'))
  x=pd.Series(xs);print('REG10',a,b,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
