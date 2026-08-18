import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 d=get_stock_daily_data(a,days=3000)
 if d is not None and len(d)>120: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); ret=p.pct_change(); vol=ret.rolling(20,min_periods=10).std()
# risk-adjusted 5-day reversal: negative recent return scaled by trailing realized volatility
f=-p.pct_change(5)/vol
obs=[]; cov=[]; turnovers=[]; decays={5:[],10:[],20:[]}
for i in range(len(p)-20):
 q=pd.concat([f.iloc[i].rename('f'),ret.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8:
  obs.append(q.f.corr(q.y));cov.append(len(q)/len(A))
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
   if len(z)>=8: turnovers.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
 for h in decays:
  q=pd.concat([f.iloc[i].rename('f'),p.pct_change(h).iloc[i+h].rename('y')],axis=1).dropna()
  if len(q)>=8: decays[h].append(q.f.corr(q.y))
x=np.asarray(obs)
print('dates',len(x),'instruments',len(px),'valid_dates',len(obs))
print('IC',np.nanmean(x),'IC_std',np.nanstd(x,ddof=1),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'coverage',np.mean(cov),'turnover',np.mean(turnovers))
for h,v in decays.items(): print('decay',h,'IC',np.nanmean(v),'dates',len(v))
print('period',p.index.min(),p.index.max())
# regime split by cross-asset median trailing volatility
for name,mask in [('early',np.arange(len(x))<len(x)//2),('late',np.arange(len(x))>=len(x)//2)]:
 z=x[mask]; print(name,'n',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1),'hit',np.mean(z>0))
print('latest coverage',f.iloc[-1].notna().sum()/len(A))
print('latest cross-sectional values',f.iloc[-1].dropna().round(4).to_dict())
