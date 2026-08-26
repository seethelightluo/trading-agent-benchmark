import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Drawdown-conditioned recovery: favor assets with a large prior drawdown,
# but only when the recent 5-session return has begun recovering; all inputs lagged.
dd=p/p.rolling(60).max()-1
rv=r.rolling(60).std(); rebound=p/p.shift(5)-1
f=(((-dd)/(rv*np.sqrt(60))) * np.maximum(rebound,0)).shift(1)
# cross-sectional IC on forward returns, minimum 8 names
stats={}
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; q=[]; ns=[]; dates=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 q=pd.Series(q,index=dates).dropna(); stats[h]=q
 print('H%d IC %.8f ICIR %.8f hit %.4f dates %d avgN %.2f'%(h,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
q=stats[10]
for n in [180,500,750]:
 z=q.iloc[-n:]; print('RECENT%d H10 IC %.8f ICIR %.8f hit %.4f dates %d'%(n,z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),len(z)))
rr=f.rank(axis=1,pct=True)
print('period',p.index.min().date(),p.index.max().date(),'rows',len(p),'assets',len(p.columns))
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),rr.diff().abs().mean(axis=1).dropna().mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20340612_drawdown_recovery_signal.csv',index=False)
print('artifact scripts/miner_1_20340612_drawdown_recovery_signal.csv')
print('H10_FINAL_IC %.10f ICIR %.10f'%(q.mean(),q.mean()/q.std(ddof=1)))
