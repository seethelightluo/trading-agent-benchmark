import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 for fn in [get_index_daily_data,get_stock_daily_data]:
  try:
   x=fn(s,6000)
   if x is not None and len(x)>0:
    x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float);break
  except Exception: pass
print('available',list(D))
p=pd.concat(D,axis=1).sort_index().ffill();r=np.log(p).diff();m=r.mean(axis=1)
def fac(i):
 v={}
 for s in D:
  rr=r[s].iloc[i-61:i];mm=m.iloc[i-61:i];ok=rr.notna()&mm.notna()
  if ok.sum()<45:continue
  b=np.cov(rr[ok],mm[ok],ddof=1)[0,1]/(np.var(mm[ok],ddof=1)+1e-12)
  z=np.log(p[s].iloc[i-1]/p[s].iloc[i-21])-b*np.log(p.iloc[i-1].mean()/p.iloc[i-21].mean());vol=rr.iloc[-20:].std(ddof=1)*np.sqrt(252)
  if vol>1e-8:v[s]=z/vol
 return v
rows=[]
for i in range(80,len(p)-10):
 v=fac(i)
 if len(v)>=8:
  z=pd.DataFrame({'f':pd.Series(v),'y':np.log(p.iloc[i+9]/p.iloc[i-1])}).dropna()
  if len(z)>=8:rows.append((p.index[i-1],len(z),z.f.corr(z.y,method='spearman')))
a=pd.DataFrame(rows,columns=['date','n','ic']).dropna();print('dates',len(a),'avgN',a.n.mean(),'coverage',a.n.sum()/(len(a)*len(D)));print('IC10 %.6f ICIR %.6f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1)*np.sqrt(len(a)),(a.ic>0).mean()));print('period',a.date.min(),a.date.max())
for n in [252,504,1008]:
 q=a.tail(n);print('recent',n,q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(len(q)))
for h in [1,5,20]:
 q=[]
 for i in range(80,len(p)-h):
  v=fac(i)
  if len(v)>=8:
   z=pd.DataFrame({'f':pd.Series(v),'y':np.log(p.iloc[i+h-1]/p.iloc[i-1])}).dropna()
   if len(z)>=8:q.append(z.f.corr(z.y,method='spearman'))
 print('decay',h,np.nanmean(q),len(q))
