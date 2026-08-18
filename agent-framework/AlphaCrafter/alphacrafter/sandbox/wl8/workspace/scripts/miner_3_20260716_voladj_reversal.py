import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None and len(d)>100:px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std()
f=-p.pct_change(3)/vol
obs=[]; turns=[]; cov=[]; dates=[]
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8: obs.append(q.f.corr(q.y));cov.append(len(q)/15);dates.append(p.index[i])
 if i:
  z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
  if len(z)>=8: turns.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
x=np.array(obs); print('dates',len(x),'n',len(px),'period',dates[0],dates[-1]); print('IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'cov',np.mean(cov),'turn',np.mean(turns))
for h in [5,10]:
 o=[]; hh=p.pct_change(h)
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),hh.iloc[i+h].rename('y')],axis=1).dropna()
  if len(q)>=8:o.append(q.f.corr(q.y))
 print('decay',h,np.nanmean(o),'n',len(o))
# regime split by cross-sectional median volatility
for label,mask in [('lowvol',vol.mean(axis=1)<=vol.mean(axis=1).rolling(252).median()),('highvol',vol.mean(axis=1)>vol.mean(axis=1).rolling(252).median())]:
 o=[]
 for i in range(len(p)-1):
  if not bool(mask.iloc[i]):continue
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8:o.append(q.f.corr(q.y))
 print(label,'IC',np.nanmean(o),'n',len(o))
