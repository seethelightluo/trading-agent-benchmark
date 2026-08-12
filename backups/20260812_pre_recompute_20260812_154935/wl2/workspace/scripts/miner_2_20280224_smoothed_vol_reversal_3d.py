import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); v=r.rolling(20,min_periods=10).std()
base=-(r)/(v+1e-5)
# Smooth the completed-bar volatility-scaled reversal over 3 sessions.
F=base.rolling(3,min_periods=3).mean()
Y={h:P.shift(-h)/P-1 for h in [1,3,5,10]}
for h in Y:
 vals=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Y[h].iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): vals.append(q);ns.append(len(z))
 x=pd.Series(vals); print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'std',round(x.std(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
rank=F.rank(axis=1,pct=True)
print('turnover',round((rank-rank.shift()).abs().mean(axis=1).mean(),6),'coverage',round(F.notna().sum().sum()/F.size,6),'avgN',round(F.notna().sum(axis=1).mean(),2),'range',P.index.min(),P.index.max())
for a in range(2020,2029):
 vals=[]
 for i in range(len(P)-1):
  if P.index[i].year==a:
   z=pd.concat([F.iloc[i],Y[1].iloc[i]],axis=1).dropna()
   if len(z)>=8:
    q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(q): vals.append(q)
 print('regime',a,'dates',len(vals),'IC',round(np.mean(vals),6) if vals else None)
