import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
a=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); q=d.sort_values('date').set_index('date').close.astype(float); a.append(q)
x=pd.concat(a,axis=1,keys=U).sort_index().values; dates=pd.concat(a,axis=1).sort_index().index
r=x[1:]/x[:-1]-1; bench=np.nanmean(r,axis=1); ics={1:[],5:[],10:[]}; ns=[]
for i in range(60,len(dates)-10):
 vals=np.full(15,np.nan)
 for j in range(15):
  rr=r[i-60:i,j]; bb=bench[i-60:i]; ok=np.isfinite(rr)&np.isfinite(bb)
  if ok.sum()<40: continue
  rr=rr[ok]; bb=bb[ok]; beta=np.cov(rr,bb,ddof=1)[0,1]/(np.var(bb,ddof=1)+1e-12)
  if not np.isfinite(x[i,j]) or not np.isfinite(x[i-20,j]):continue
  vals[j]=(x[i,j]/x[i-20,j]-1-beta*(np.prod(1+bench[max(0,i-20):i])-1))/(np.std(rr-beta*bb)*np.sqrt(252)+1e-8)
 for h in ics:
  if i+h>=len(r):continue
  f=vals; y=np.prod(1+r[i+1:i+h+1],axis=0)-1; ok=np.isfinite(f)&np.isfinite(y)
  if ok.sum()>=8: ics[h].append(spearmanr(f[ok],y[ok]).statistic)
 ns.append(np.isfinite(vals).sum())
for h,v in ics.items():
 z=np.array(v); print('h',h,'dates',len(z),'mean_n',np.mean(ns),'IC',np.mean(z),'ICIR',np.mean(z)/(np.std(z,ddof=1)+1e-12),'hit',np.mean(z>0))
print('coverage',np.mean(ns)/15)
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 v=[]
 for i in range(60,len(dates)-1):
  if lo<=dates[i].year<=hi:
   # omitted detailed split; use daily recomputation not needed
   pass
