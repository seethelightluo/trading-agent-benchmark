import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 try:
  x=get_stock_daily_data(a,days=4000)
  if x is not None: D[a]=x.set_index('date').close.astype(float)
 except Exception as e: pass
p=pd.concat(D,axis=1,sort=True).ffill(); r=p.pct_change()
# Low realized volatility, with a volatility-trend adjustment; exact completed-day signal.
for lb in [5,10,20,40]:
 f=-r.rolling(lb,min_periods=max(3,lb//2)).std()
 vals=[]; dates=[]; nms=[]
 for i,dt in enumerate(p.index[:-1]):
  q=pd.concat([f.loc[dt].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8: vals.append(q.f.corr(q.y));dates.append(dt);nms.append(len(q))
 s=pd.Series(vals); print('LOWVOL',lb,'dates',len(s),'avgN',round(np.mean(nms),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4),'cov',round(np.mean(nms)/15,4))
# vol contraction: prior short vol relative long vol, favor compressed assets
f=-(r.rolling(5).std()/r.rolling(30).std())
vals=[];nms=[]
for i,dt in enumerate(p.index[:-1]):
 q=pd.concat([f.loc[dt].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8: vals.append(q.f.corr(q.y));nms.append(len(q))
s=pd.Series(vals);print('CONTRACTION dates',len(s),'avgN',round(np.mean(nms),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4),'cov',round(np.mean(nms)/15,4))
# regime breakdown contraction
print('years',[(y,round(s[[d.year==y for d in dates]].mean(),5)) for y in sorted(set(d.year for d in dates))] if False else '')
