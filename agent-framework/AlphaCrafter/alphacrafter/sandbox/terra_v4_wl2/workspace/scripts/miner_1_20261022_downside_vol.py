import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; A=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv').sort_values('date'); x['date']=pd.to_datetime(x.date); A.append(x.set_index('date').close)
P=pd.concat(A,axis=1,join='inner').values; R=P[1:]/P[:-1]-1; n=len(P)
for h in [1,5,10]:
 q=[]; ns=[]; tr=[]; prev=None
 for i in range(25,n-h-1):
  rr=R[i-20:i]; down=np.sqrt(np.nanmean(np.minimum(rr,0)**2,axis=0)); f=-down; y=P[i+h]/P[i]-1; ok=np.isfinite(f)&np.isfinite(y)&(down>0)
  if ok.sum()>=8:
   q.append(spearmanr(f[ok],y[ok]).statistic);ns.append(ok.sum()); cur=np.full(15,np.nan);cur[ok]=pd.Series(f[ok]).rank().values
   if prev is not None: tr.append(np.nanmean(cur[ok]!=prev[ok]))
   prev=cur
 q=np.array(q); print('h',h,'dates',len(q),'avg',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4),'turn',round(np.mean(tr),4))
 for lo,hi in [(0,758),(758,1274),(1274,9999)]:
  z=q[lo:hi]
  if len(z)>1: print(' regime',len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('dates',n,'instruments',15)
