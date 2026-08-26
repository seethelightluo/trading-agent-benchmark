import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Candidate: short-term shock reversal only in an established, still-rising trend.
# The dual trend gate avoids reversal calls during persistent bear regimes.
base=-(p/p.shift(5)-1)/(r.rolling(40).std()*np.sqrt(5))
f=base.where((p/p.shift(20)-1>0)&(p/p.shift(60)-1>0),0.0).shift(1)
print('period',p.index.min().date(),p.index.max().date(),'rows',len(p),'assets',len(p.columns))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; q=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(q).dropna(); print(h,'IC %.8f ICIR %.8f hit %.4f dates %d avgN %.2f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20340403_dualtrend_reversal_signal.csv',index=False)
