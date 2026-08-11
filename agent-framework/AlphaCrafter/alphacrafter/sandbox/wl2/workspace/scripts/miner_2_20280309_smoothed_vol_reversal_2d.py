import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): D[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(D).sort_index(); r=P.pct_change(); v=r.rolling(20,min_periods=15).std()
# Two-session smoothed volatility-scaled reversal; all inputs are completed bars.
F=-(r/(v+1e-5)).rolling(2,min_periods=2).mean()
Y={h:P.shift(-h)/P-1 for h in [1,3,5,10]}
for h,y in Y.items():
 vals=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): vals.append(q);ns.append(len(z))
 x=pd.Series(vals); print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'std',round(x.std(ddof=1),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
rank=F.rank(axis=1,pct=True)
print('turnover',round(rank.diff().abs().mean(axis=1).mean(),6),'coverage',round(F.notna().sum().sum()/F.size,6),'avgN',round(F.notna().sum(axis=1).mean(),2),'instruments',len(D),'period',P.index.min(),P.index.max())
y=Y[1];
for yr in range(2020,2029):
 a=[]
 for i in range(len(P)-1):
  if P.index[i].year==yr:
   z=pd.concat([F.iloc[i],y.iloc[i]],axis=1).dropna()
   if len(z)>=8:
    q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(q):a.append(q)
 print('regime',yr,'dates',len(a),'IC',round(np.mean(a),6) if a else None)
