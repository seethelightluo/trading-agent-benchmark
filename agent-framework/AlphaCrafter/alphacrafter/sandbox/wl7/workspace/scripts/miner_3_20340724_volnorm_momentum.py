import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date')['close'].astype(float)
P=pd.concat(D,axis=1).sort_index().ffill()
ret20=P/P.shift(20)-1; vol20=P.pct_change().rolling(20,min_periods=15).std()*np.sqrt(20)
F=(ret20/vol20.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).shift(1)
for h in [1,5,10,20]:
 R=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q); ns.append(len(z))
 q=pd.Series(vals); print('H%d IC %.8f ICIR %.8f hit %.4f dates %d avgN %.2f'%(h,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
 if h==10:
  for n in [180,500,750]:
   z=q.iloc[-n:]; print('recent%d IC %.8f ICIR %.8f hit %.4f dates %d'%(n,z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),len(z)))
r=F.rank(axis=1,pct=True)
print('period',P.index.min().date(),P.index.max().date(),'rows',len(P),'assets',len(P.columns),'coverage',F.notna().mean().mean(),'rankturn',r.diff().abs().mean(axis=1).dropna().mean())
out=F.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20340724_volnorm_momentum_signal.csv',index=False); print('artifact rows',len(out))
