import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Low-volatility quality: inverse realized volatility, mildly conditioned on positive trailing return.
vol=r.rolling(30,min_periods=20).std()
trend=P/P.shift(20)-1
F=(1/(vol+1e-6))*(1+0.25*np.tanh(trend/0.1))
for h in [1,3,5,10]:
 Y=P.shift(-h)/P-1; vals=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman')); ns.append(len(z))
 x=pd.Series(vals).dropna(); print('h',h,'dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'std',x.std(),'ICIR',x.mean()/x.std(),'hit',(x>0).mean())
rank=F.rank(axis=1,pct=True); print('turnover', (rank-rank.shift()).abs().mean(axis=1).mean())
print('range',P.index.min(),P.index.max())
for a,b in [('2026','2026'),('2027','2027'),('2028','2028')]:
 vals=[]
 Y=P.shift(-1)/P-1
 for i in range(len(P)-1):
  if str(P.index[i].year) not in [a]: continue
  z=pd.concat([F.iloc[i],Y.iloc[i]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(a,len(vals),np.mean(vals) if vals else np.nan)
