import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Volatility-normalized medium-term momentum: recent 20d return divided by
# 20d realized volatility, rewarding persistent gains per unit risk.
vol=r.rolling(20,min_periods=10).std()
F=(P/P.shift(20)-1)/(vol*np.sqrt(20)+1e-5)
for h in [1,3,5,10]:
 vals=[]; ns=[]; Y=P.shift(-h)/P-1
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): vals.append(q); ns.append(len(z))
 x=pd.Series(vals)
 print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'std',round(x.std(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
rank=F.rank(axis=1,pct=True)
print('turnover',round((rank-rank.shift()).abs().mean(axis=1).mean(),6),'coverage',round(F.notna().sum().sum()/F.size,6),'avgN',round(F.notna().sum(axis=1).mean(),2),'range',P.index.min(),P.index.max())
for a in range(2020,2029):
 vals=[]; Y=P.shift(-1)/P-1
 for i in range(len(P)-1):
  if P.index[i].year!=a: continue
  z=pd.concat([F.iloc[i],Y.iloc[i]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q)
 print('regime',a,'dates',len(vals),'IC',round(np.mean(vals),6) if vals else None)
