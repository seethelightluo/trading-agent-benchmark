import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); m=r.mean(axis=1)
# Market-residual short-term reversal: reverse cumulative 5d return unexplained by common cross-asset market move.
beta=r.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0)
res=r.sub(beta.mul(m,axis=0),axis=0)
res5=res.rolling(5,min_periods=5).sum(); rv=res.rolling(20,min_periods=15).std()
F=-res5/(rv+1e-6)
for h in [1,3,5,10]:
 Y=P.shift(-h)/P-1; vals=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman')); ns.append(len(z))
 x=pd.Series(vals).dropna(); print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'std',round(x.std(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
print('turnover',round((F.rank(axis=1,pct=True)-F.rank(axis=1,pct=True).shift()).abs().mean(axis=1).mean(),6),'coverage',round(F.notna().mean().mean(),4))
for yr in range(2020,2029):
 vals=[]
 Y=P.shift(-1)/P-1
 for i in range(len(P)-1):
  if P.index[i].year==yr:
   z=pd.concat([F.iloc[i],Y.iloc[i]],axis=1).dropna()
   if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 if vals: print('regime',yr,len(vals),round(np.mean(vals),6))
