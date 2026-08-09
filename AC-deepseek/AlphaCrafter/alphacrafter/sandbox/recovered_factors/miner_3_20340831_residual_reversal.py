import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(p):
 d=pd.read_csv(p,parse_dates=['date']); return d.set_index(d.date.dt.normalize()).close
P=pd.DataFrame({a:ld('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index(); R=np.log(P).diff(); m=R.mean(axis=1)
for w in [3,5,10]:
 beta=pd.DataFrame(index=R.index,columns=A,dtype=float)
 mv=m.rolling(40,min_periods=20).var()
 for a in A: beta[a]=R[a].rolling(40,min_periods=20).cov(m)/mv
 resid=R-beta.multiply(m,axis=0)
 sig=-resid.rolling(w,min_periods=w).sum()/R.rolling(20,min_periods=15).std()
 f=R.shift(-1); z=[]; ds=[]; ns=[]
 for d in P.index:
  x,y=sig.loc[d],f.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
   z.append(spearmanr(x[ok],y[ok]).statistic);ds.append(d);ns.append(ok.sum())
 z=np.array(z); ds=pd.DatetimeIndex(ds)
 print('W',w,'dates',len(z),'N',np.mean(ns),'cov',sig.notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
 for lo,hi in [('2020','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
  q=z[(ds>=lo+'-01-01')&(ds<=hi+'-12-31')]; print(lo,round(q.mean(),4),round(q.mean()/q.std(ddof=1),4) if len(q)>1 else np.nan,len(q))
