import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None and len(d)>100: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change(); r3=p.pct_change(3); f=(-r3).where(r3<0)
def run(start,end):
 start,end=pd.Timestamp(start),pd.Timestamp(end); ic=[];cov=[]
 for i in range(len(p)-1):
  if not(start<=p.index[i]<=end): continue
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: ic.append(q.f.corr(q.y));cov.append(len(q)/15)
 x=np.array(ic); return len(x),np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1),np.mean(x>0),np.mean(cov)
print('n_assets',len(px),'dates',p.index.min(),p.index.max())
for s,e in [('2020-01-01','2022-02-28'),('2022-03-01','2024-04-30'),('2024-05-01','2026-07-15')]: print(s,e,run(s,e))
tr=[]
for i in range(1,len(p)):
 z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
 if len(z)>=8: tr.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
print('turnover',np.mean(tr),len(tr))
for h in [5,10]:
 o=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),p.pct_change(h).iloc[i+h].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:o.append(q.f.corr(q.y))
 print('decay',h,len(o),np.nanmean(o))
libs={'rev3':-r3,'rev5':-p.pct_change(5),'ram20':p.pct_change(20)/(p.pct_change().rolling(20).std()*np.sqrt(20))}
for n,x in libs.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); print('corr',n,z.f.corr(z.x),len(z))
print('coverage',f.notna().sum().sum()/f.size)
