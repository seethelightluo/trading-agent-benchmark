import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None and len(d)>100:px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); ret=p.pct_change(); r3=p.pct_change(3)
for name,f in [('loss_only',(-r3).where(r3<0)),('gain_only',(-r3).where(r3>=0)),('tail_clip',-r3.clip(-.08,.08))]:
 obs=[];turn=[]; cov=[]
 for i in range(len(p)-1):
  q=pd.concat([f.iloc[i].rename('f'),ret.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: obs.append(q.f.corr(q.y));cov.append(len(q)/15)
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:turn.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
 x=np.array(obs); print(name,'dates',len(x),'n',len(px),'IC',round(np.nanmean(x),5),'ICIR',round(np.nanmean(x)/np.nanstd(x,ddof=1),5),'hit',round(np.mean(x>0),5),'cov',round(np.mean(cov),5),'turn',round(np.mean(turn),5))
 for h in [5,10]:
  oo=[]
  for i in range(len(p)-h):
   q=pd.concat([f.iloc[i].rename('f'),p.pct_change(h).iloc[i+h].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1:oo.append(q.f.corr(q.y))
  print(' decay',h,round(np.nanmean(oo),5),len(oo))
