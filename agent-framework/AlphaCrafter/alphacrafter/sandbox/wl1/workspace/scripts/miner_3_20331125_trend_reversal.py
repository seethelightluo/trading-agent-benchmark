import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:
  x=get_stock_daily_data(s,days=5000)
  if x is not None and len(x): x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
 except: pass
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); v=r.rolling(20).std(); trend=p.pct_change(40)
# short reversal, only when medium trend is positive; otherwise neutral
f=(-p.pct_change(5)/(v*np.sqrt(5)+1e-8)*((trend>0).astype(float))).shift(1); y=p.pct_change(10).shift(-10); rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mu=z.ic.mean(); sd=z.ic.std()
print('assets',len(D),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15); print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(mu,mu/sd,(z.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for label,lo,hi in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033','2033-01-01','2033-11-25')]:
 q=z.loc[lo:hi].ic; print(label,len(q),q.mean(),q.mean()/q.std())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20331125_trend_reversal_signal.csv',index=False)
