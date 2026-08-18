import pandas as pd,numpy as np
from scipy.stats import rankdata
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
A=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',usecols=['date','close'],parse_dates=['date']).set_index('date')['close'].sort_index(); A.append(x)
p=pd.concat(A,axis=1,keys=U).sort_index().to_numpy(float); r=np.diff(np.log(p),axis=0); dates=pd.concat(A,axis=1).sort_index().index[1:]
# 60d beta-neutral residual momentum, 20d residual return, 10d forward return
W,L,H=60,20,10; ics=[]; ns=[]; by=[]
for i in range(W,len(r)-H):
 z=r[i-W:i]; b=np.nanmean(z,axis=1); out=[]; y=[]
 for j in range(15):
  ok=np.isfinite(z[:,j])&np.isfinite(b)
  if ok.sum()<30: out.append(np.nan); y.append(np.nan); continue
  zz=z[ok,j]; bb=b[ok]; beta=np.cov(zz,bb,ddof=1)[0,1]/(np.var(bb,ddof=1)+1e-12)
  resid=zz-beta*bb; out.append(resid[-L:].sum()); y.append(np.nansum(r[i+1:i+1+H,j]))
 ok=np.isfinite(out)&np.isfinite(y)
 if ok.sum()>=8:
  # Spearman without scipy call
  x=rankdata(np.asarray(out)[ok]); q=rankdata(np.asarray(y)[ok]); ics.append(np.corrcoef(x,q)[0,1]); ns.append(ok.sum()); by.append((dates[i],ics[-1]))
a=np.array(ics); print('factor residual beta-neutral momentum; dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),5),'hit',round(np.mean(a>0),3),'coverage',round(np.mean(ns)/15,4))
for n in [250,500,1000]:
 q=a[-n:]; print('recent',n,'dates',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/(q.std(ddof=1)+1e-12),5),'hit',round(np.mean(q>0),3))
