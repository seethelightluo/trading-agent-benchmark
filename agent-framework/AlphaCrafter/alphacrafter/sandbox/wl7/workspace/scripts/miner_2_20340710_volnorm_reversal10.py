import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5200)
 if x is not None and len(x): D[s]=x.set_index(pd.to_datetime(x.date)).close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# lagged 10d reversal scaled by trailing 20d realized volatility; all inputs end before forecast date
f=(-p.pct_change(10)/r.rolling(20).std()).shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; q=[]; ns=[]; ds=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(q,index=ds).dropna(); print(f'H{h} IC {q.mean():.8f} ICIR {q.mean()/q.std(ddof=1):.8f} hit {(q>0).mean():.4f} dates {len(q)} avgN {np.mean(ns):.2f}')
print('period',p.index.min().date(),p.index.max().date(),'rows',len(p),'assets',len(p.columns))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
o=f.stack().rename('signal').reset_index();o.columns=['date','symbol','signal'];o.to_csv('scripts/miner_2_20340710_volnorm_reversal10_signal.csv',index=False);print('artifact scripts/miner_2_20340710_volnorm_reversal10_signal.csv');print('max_abs_library_correlation null')
