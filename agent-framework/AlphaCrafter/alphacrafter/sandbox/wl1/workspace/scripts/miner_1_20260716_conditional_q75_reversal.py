import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 d=get_stock_daily_data(a,days=3000)
 if d is not None and len(d)>120: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change(); v=r.rolling(20,min_periods=10).std()
# Conditional short reversal, retaining the calmer 75% of the cross-section
f=(-p.pct_change(3)).where(v.le(v.quantile(.75,axis=1),axis=0)); obs=[]; cov=[]; turn=[]; d5=[]
for i in range(len(p)-5):
 q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
  c=q.f.corr(q.y); obs.append(c);cov.append(len(q)/len(A))
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
   if len(z)>=8: turn.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
  q2=pd.concat([f.iloc[i].rename('f'),p.pct_change(5).iloc[i+5].rename('y')],axis=1).dropna()
  if len(q2)>=8 and q2.f.nunique()>1 and q2.y.nunique()>1:d5.append(q2.f.corr(q2.y))
x=np.asarray(obs);print('factor=conditional_3d_reversal_vol_q75');print('dates',len(x),'instruments',len(px),'IC',x.mean(),'std',x.std(ddof=1),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'coverage',np.mean(cov),'turnover',np.mean(turn),'decay5',np.mean(d5),'period',p.index.min().date(),p.index.max().date())
for label,m in [('early',np.arange(len(x))<len(x)//2),('late',np.arange(len(x))>=len(x)//2)]:
 z=x[m];print(label,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
