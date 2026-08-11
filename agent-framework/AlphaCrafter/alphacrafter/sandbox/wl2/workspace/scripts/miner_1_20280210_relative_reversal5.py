import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Relative short-horizon reversal: negate 5-session return relative to the
# contemporaneous cross-sectional median, scaled by recent volatility.
rel=P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0)
vol=r.rolling(20,min_periods=15).std()
F=-rel/(vol+1e-6); rank=F.rank(axis=1,pct=True)
for h in [1,3,5,10]:
 vals=[];ns=[];Y=P.shift(-h)/P-1
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:
   c=z.f.corr(z.y,method='spearman')
   if pd.notna(c):vals.append(c);ns.append(len(z))
 x=np.array(vals);print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'std',round(x.std(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round(np.mean(x>0),4))
print('turnover',round((rank-rank.shift()).abs().mean(axis=1).mean(),6),'avg coverage',round(F.notna().sum(axis=1).mean()/15,4),'range',P.index.min(),P.index.max())
for yr in [2026,2027,2028]:
 vals=[];Y=P.shift(-1)/P-1
 for i in range(len(P)-1):
  if P.index[i].year==yr:
   z=pd.concat([F.iloc[i],Y.iloc[i]],axis=1).dropna()
   if len(z)>=8:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(c):vals.append(c)
 print('regime',yr,'dates',len(vals),'IC',round(np.mean(vals),6) if vals else None,'hit',round(np.mean(np.array(vals)>0),4) if vals else None)
