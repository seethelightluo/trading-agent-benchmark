import numpy as np,pandas as pd,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in A:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close
P=pd.concat(D,axis=1).sort_index().ffill(); R=P.pct_change().values; M=np.nanmean(R,axis=1); n,k=R.shape
for weights in [(1,0,0),(0,1,0),(0,0,1),(2,1,0),(1,2,0),(2,2,1)]:
 ic=[]; ns=[]
 for i in range(130,n-1):
  disp=np.nanstd(R[i-9:i+1],axis=1).mean(); hist=np.array([np.nanstd(R[j-9:j+1],axis=1).mean() for j in range(max(10,i-119),i+1)])
  if disp<=np.nanmedian(hist): continue
  f=np.zeros(k)
  for c in range(k):
   rr=R[i-29:i+1,c]; mm=M[i-29:i+1]; ok=np.isfinite(rr)&np.isfinite(mm)
   beta=np.cov(rr[ok],mm[ok],ddof=1)[0,1]/(np.var(mm[ok],ddof=1)+1e-8); res=rr-beta*mm; sd=np.nanstd(rr)+1e-6
   f[c]=sum(w*(-np.nansum(res[-h:])/sd) for w,h in zip(weights,(3,5,10)))
  ok=np.isfinite(f)&np.isfinite(R[i+1]);
  if ok.sum()>=8: ic.append(spearmanr(f[ok],R[i+1,ok]).statistic); ns.append(ok.sum())
 x=np.array(ic); print(weights,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
