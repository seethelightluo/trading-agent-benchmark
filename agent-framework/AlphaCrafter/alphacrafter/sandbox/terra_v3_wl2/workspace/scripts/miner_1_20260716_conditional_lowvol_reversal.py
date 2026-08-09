import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for a in A:
 d=get_stock_daily_data(a,days=3000)
 if d is not None and len(d)>120:px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change(); v=r.rolling(20,min_periods=10).std(); raw=-p.pct_change(3)
f=raw.where(v.le(v.median(axis=1),axis=0)); obs=[];cov=[];turn=[];dec=[]
def corr(q):
 if len(q)<8 or q.nunique().min()<2:return np.nan
 return q.iloc[:,0].corr(q.iloc[:,1])
for i in range(len(p)-5):
 q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna(); c=corr(q)
 if np.isfinite(c):
  obs.append(c);cov.append(len(q)/len(A))
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
   if len(z)>=8:turn.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
  q2=pd.concat([f.iloc[i].rename('f'),p.pct_change(5).iloc[i+5].rename('y')],axis=1).dropna(); c2=corr(q2)
  if np.isfinite(c2):dec.append(c2)
x=np.asarray(obs);print('dates',len(x),'instruments',len(px),'IC',x.mean(),'std',x.std(ddof=1),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'coverage',np.mean(cov),'turnover',np.mean(turn),'decay5',np.mean(dec),'period',p.index.min(),p.index.max())
for label,m in [('early',np.arange(len(x))<len(x)//2),('late',np.arange(len(x))>=len(x)//2)]:
 z=x[m];print(label,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
