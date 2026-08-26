import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); ret60=p.pct_change(60); breadth=ret60.median(axis=1)
# Correct row-wise demeaning: subtract Series along index (axis=0).
f=ret60.sub(breadth,axis=0).mul(np.sign(breadth),axis=0).shift(1)
res={}
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; q=[]; ns=[]; dates=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 q=pd.Series(q,index=dates).dropna(); res[h]=q
 print('H%d IC %.8f ICIR %.8f hit %.4f dates %d avgN %.2f'%(h,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
for n in [180,500,750]:
 q=res[10].iloc[-n:]
 print('recent%d H10 IC %.8f ICIR %.8f hit %.4f dates %d'%(n,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
rr=f.rank(axis=1,pct=True)
print('period',p.index.min().date(),p.index.max().date(),'rows',len(p),'assets',len(p.columns))
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),rr.diff().abs().mean(axis=1).dropna().mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340612_breadth_conditioned_trend_signal.csv',index=False)
print('artifact scripts/miner_2_20340612_breadth_conditioned_trend_signal.csv')
print('max_abs_library_correlation null')
