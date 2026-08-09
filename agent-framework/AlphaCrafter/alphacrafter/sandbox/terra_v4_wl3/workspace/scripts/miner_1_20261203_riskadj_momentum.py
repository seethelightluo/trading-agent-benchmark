import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-12-03'
A={}; dates=None
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).query("date<=@end").sort_values('date')
 A[s]=(d.date.to_numpy(),d.close.to_numpy(float)); dates=set(d.date)
# use intersection to eliminate alignment costs
dates=sorted(dates)
rows=[]; artifacts=[]
for dt in dates[25:-1]:
 vs=[]; ys=[]; syms=[]
 for s,(ds,c) in A.items():
  ix=np.searchsorted(ds,dt)
  if ix>=len(ds) or ds[ix]!=dt or ix<21 or ix+1>=len(c):continue
  r=np.diff(c[max(0,ix-20):ix+1])/c[max(0,ix-20):ix]; f=(c[ix]/c[ix-20]-1)/(np.std(r)+1e-8); y=c[ix+1]/c[ix]-1
  if np.isfinite(f) and np.isfinite(y):vs.append(f);ys.append(y);syms.append(s)
 if len(vs)>=8:
  rows.append((dt,spearmanr(vs,ys).statistic,len(vs)))
  artifacts += [(dt,s,f) for s,f in zip(syms,vs)]
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,z in [('all',x),('2020-22',x.loc['2020':'2022']),('2023-24',x.loc['2023':'2024']),('2025-26',x.loc['2025':'2026'])]:
 print(label,'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(z.ic.mean(),5),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4))
print('coverage',round(x.n.sum()/(len(x)*15),4))
for h in [1,5,10]:
 rr=[]
 for dt in x.index:
  vs=[];ys=[]
  for s,(ds,c) in A.items():
   ix=np.searchsorted(ds,dt)
   if ix>=len(ds) or ds[ix]!=dt or ix<21 or ix+h>=len(c):continue
   r=np.diff(c[ix-20:ix+1])/c[ix-20:ix]; vs.append((c[ix]/c[ix-20]-1)/(np.std(r)+1e-8));ys.append(c[ix+h]/c[ix]-1)
  if len(vs)>=8:rr.append(spearmanr(vs,ys).statistic)
 print('h',h,'IC',round(np.mean(rr),5),'ICIR',round(np.mean(rr)/np.std(rr,ddof=1),5),'dates',len(rr))
pd.DataFrame(artifacts,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20261203_riskadj_momentum_signal.csv',index=False)
