import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Candidate: volatility-normalized 60-session reversal, lagged one day.
f=(-(p/p.shift(60)-1)/(r.rolling(120).std()*np.sqrt(60))).shift(1)
yields={}
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; q=[]; ns=[]; dates=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 q=pd.Series(q,index=dates).dropna(); yields[h]=q
 print('H%d IC %.8f ICIR %.8f hit %.4f dates %d avgN %.2f'%(h,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
for label, n in [('full',len(p)),('recent180',180),('recent500',500),('recent750',750)]:
 q=yields[10]
 if label!='full': q=q.iloc[-n:]
 print(label,'H10 IC %.8f ICIR %.8f hit %.4f dates %d'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
rr=f.rank(axis=1,pct=True)
print('period',p.index.min().date(),p.index.max().date(),'rows',len(p),'assets',len(p.columns))
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),rr.diff().abs().mean(axis=1).dropna().mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20340515_medium_reversal60_signal.csv',index=False)
# pairwise artifact rank correlation against prior variants when available
for fn in ['scripts/miner_3_20340417_medium_reversal30_signal.csv','scripts/miner_3_20340501_medium_reversal45_signal.csv']:
 try:
  a=pd.read_csv(fn); a.date=pd.to_datetime(a.date); a=a.pivot(index='date',columns='symbol',values='signal')
  print('corr',fn, f.stack().rename('new').to_frame().join(a.stack().rename('old'),how='inner').corr().iloc[0,1])
 except Exception as e: print('corr_error',fn,str(e))
