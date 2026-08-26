import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Breadth-conditioned risk-adjusted momentum: use 20d momentum only when breadth confirms,
# otherwise reverse the 10d move; all inputs lagged one day.
m20=p/p.shift(20)-1; m10=p/p.shift(10)-1; v40=r.rolling(40).std()*np.sqrt(20)
breadth=(m20>0).mean(axis=1)
f=(m20.where(breadth>=0.5,-m10)/v40).shift(1)
res={}
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; q=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(q).dropna(); res[h]={'ic':float(q.mean()),'icir':float(q.mean()/q.std(ddof=1)),'hit':float((q>0).mean()),'dates':len(q),'avg_n':float(np.mean(ns))}
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('period',p.index.min(),p.index.max(),'assets',len(p.columns),'rows',len(p))
print(res); print('coverage',float(f.notna().mean().mean()),'turnover',float(turn))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340417_breadth_confirmed_momentum_signal.csv',index=False)
# recent windows
for n in [180,500,750]:
 ds=p.index[-n:]; vals=[]
 for dt in ds:
  z=pd.concat([f.loc[dt],(p.shift(-10)/p-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals).dropna(); print('recent',n,'ic',q.mean(),'icir',q.mean()/q.std(ddof=1),'dates',len(q))
