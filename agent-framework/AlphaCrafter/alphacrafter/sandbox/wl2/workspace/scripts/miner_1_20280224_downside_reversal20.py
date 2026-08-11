import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Downside-risk adjusted intermediate reversal: reverse 20d relative return,
# normalized by 30d downside deviation. Cross-sectional median neutralization.
down=r.where(r<0).rolling(30,min_periods=15).std()
ret20=P/P.shift(20)-1
rel=ret20.sub(ret20.median(axis=1),axis=0)
F=-rel/(down+1e-5)
Ybase=P.shift(-1)/P-1
for h in [1,3,5,10]:
 Y=P.shift(-h)/P-1; vals=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman')); ns.append(len(z))
 x=pd.Series(vals).dropna(); print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'std',round(x.std(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
rank=F.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift()).abs().mean(axis=1).mean(),6),'coverage',round(F.notna().mean().mean(),4),'range',P.index.min(),P.index.max())
for yr in [2020,2021,2022,2023,2024,2025,2026,2027,2028]:
 vals=[]
 for i in range(len(P)-1):
  if P.index[i].year!=yr: continue
  z=pd.concat([F.iloc[i],Ybase.iloc[i]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 if vals: print('regime',yr,'dates',len(vals),'IC',round(np.mean(vals),6),'ICIR',round(np.mean(vals)/np.std(vals),6))
