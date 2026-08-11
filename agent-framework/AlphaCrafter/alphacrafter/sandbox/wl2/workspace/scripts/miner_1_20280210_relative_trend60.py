import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Cross-asset relative trend: 60d return minus contemporaneous cross-sectional median,
# normalized by 30d volatility. All inputs are lagged at decision date.
rel=P.pct_change(60).sub(P.pct_change(60).median(axis=1),axis=0)
vol=r.rolling(30,min_periods=20).std()
F=rel/(vol+1e-6)
Y=P.shift(-1)/P-1
for h in [1,3,5,10]:
 vals=[]; ns=[]
 Yh=P.shift(-h)/P-1
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Yh.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:
   c=z.f.corr(z.y,method='spearman')
   if pd.notna(c): vals.append(c);ns.append(len(z))
 x=pd.Series(vals); print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'std',round(x.std(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
rank=F.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift()).abs().mean(axis=1).mean(),6),'coverage',round(F.notna().sum(axis=1).mean()/15,4))
print('range',P.index.min(),P.index.max())
for yr in [2020,2021,2022,2023,2024,2025,2026,2027,2028]:
 vals=[]; Y1=P.shift(-1)/P-1
 for i in range(len(P)-1):
  if P.index[i].year!=yr: continue
  z=pd.concat([F.iloc[i],Y1.iloc[i]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c)
 print('regime',yr,'dates',len(vals),'IC',round(np.mean(vals),6) if vals else None,'hit',round(np.mean(np.array(vals)>0),4) if vals else None)
