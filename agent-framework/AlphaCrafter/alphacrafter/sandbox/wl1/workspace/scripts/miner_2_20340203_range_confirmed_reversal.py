import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Range-confirmed reversal: recent pullback, risk normalized, enhanced when closes recover toward daily highs.
clv=((p*0 + 1))
# reload high/low
H={};L={}
for s in D:
 x=get_stock_daily_data(s,days=5000).assign(date=lambda q:pd.to_datetime(q.date)).set_index('date'); H[s]=x.high; L[s]=x.low
high=pd.DataFrame(H).reindex(p.index).ffill(); low=pd.DataFrame(L).reindex(p.index).ffill()
clv=((p-low)/(high-low).replace(0,np.nan)-.5).rolling(5).mean()
f=(-p.pct_change(5)/(r.rolling(20).std()+1e-8)*(1+clv)).shift(1)
rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],p.pct_change(10).shift(-10).loc[dt]],axis=1).dropna()
 if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mu=z.ic.mean(); sd=z.ic.std()
print('assets',len(D),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15)
print('10d IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(mu,mu/sd,(z.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for label,lo,hi in [('2020-23','2020-01-01','2023-12-31'),('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033-34','2033-01-01','2034-12-31')]:
 q=z.loc[lo:hi].ic; print(label,len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
for h in [1,5,10,20]:
 yy=p.pct_change(h).shift(-h); q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1]))
 print('decay',h,np.nanmean(q),np.nanmean(q)/np.nanstd(q))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340203_range_confirmed_reversal_signal.csv',index=False)
