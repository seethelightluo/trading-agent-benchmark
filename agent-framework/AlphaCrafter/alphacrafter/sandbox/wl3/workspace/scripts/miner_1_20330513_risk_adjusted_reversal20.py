import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is None or len(x)==0:
  try:x=get_index_daily_data(s,4000)
  except FileNotFoundError:x=None
 if x is not None and len(x):
  z=x[['date','close']].drop_duplicates('date').set_index('date').close.astype(float);D[s]=z
px=pd.DataFrame(D).sort_index(); lr=np.log(px).diff(); v=lr.rolling(20,min_periods=15).std()*np.sqrt(252)
sig=(-px.pct_change(20)/v); f=px.shift(-10)/px-1
rows=[]
for d in sig.index:
 a=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
 if len(a)>=8:
  q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if pd.notna(q):rows.append((d,q,len(a)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=risk_adjusted_reversal20; dates=%d avg_n=%.2f coverage=%.3f'%(len(r),r.n.mean(),r.n.sum()/(len(r)*15)))
print('IC=%.6f ICIR=%.6f hit=%.3f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1),(r.ic>0).mean()))
for h in [1,3,5,10,20]:
 f=px.shift(-h)/px-1; q=[]
 for d in sig.index:
  a=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(a)>=8:
   z=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(z):q.append(z)
 print('decay_%d IC %.6f n %d'%(h,np.mean(q),len(q)))
for n in [120,252,756]:
 q=r.tail(n).ic;print('recent',n,'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('turnover=%.6f'%sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
sig.to_csv('scripts/miner_1_20330513_risk_adjusted_reversal20_signal.csv');r.to_csv('scripts/miner_1_20330513_risk_adjusted_reversal20_ic.csv')
