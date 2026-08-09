import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
A=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv').sort_values('date'); x['date']=pd.to_datetime(x.date); A.append(x.set_index('date').close)
P=pd.concat(A,axis=1,join='inner').values; dates=len(P); R=P[1:]/P[:-1]-1
for h in [1,5,10]:
  ics=[]; ns=[]; turns=[]; prev=None
  for i in range(25,dates-h-1):
    v5=np.std(R[i-5:i],axis=0); v20=np.std(R[i-25:i-5],axis=0); f=-np.log((v5+1e-10)/(v20+1e-10)); y=P[i+h]/P[i]-1
    ok=np.isfinite(f)&np.isfinite(y)&(v20>0)
    if ok.sum()>=8:
      ics.append(spearmanr(f[ok],y[ok]).statistic); ns.append(ok.sum()); cur=np.full(15,np.nan); cur[ok]=pd.Series(f[ok]).rank().values
      if prev is not None: turns.append(np.nanmean(cur[ok]!=prev[ok]))
      prev=cur
  q=np.array(ics); print('h',h,'dates',len(q),'avg_names',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4),'turn',round(np.mean(turns),4))
  for lo,hi in [(0,758),(758,1274),(1274,9999)]:
   z=q[lo:hi]
   if len(z)>1: print(' regime',len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('dates',dates,'instruments',len(U))
