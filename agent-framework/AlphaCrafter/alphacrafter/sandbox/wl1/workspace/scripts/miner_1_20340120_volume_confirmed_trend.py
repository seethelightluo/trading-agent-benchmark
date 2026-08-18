import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={};V={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  q=x.assign(date=pd.to_datetime(x.date)).set_index('date');C[s]=q.close.astype(float);V[s]=q.volume.astype(float)
p=pd.DataFrame(C).sort_index().ffill(); vol=pd.DataFrame(V).reindex(p.index).ffill(); r=p.pct_change()
# Volume-confirmed medium trend: 20d volatility-scaled return, amplified by persistent relative volume.
rv=r.rolling(20).std(); vr=(vol.rolling(20).mean()/(vol.rolling(60).mean()+1e-12)).clip(.5,2.0)
f=(r.rolling(20).sum()/(rv+1e-8)*vr).shift(1); y=p.pct_change(10).shift(-10); rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('assets',len(C),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15)
print('dailyIC %.8f dailyICIR %.8f hit %.4f turnover %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for label,lo,hi in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033-34','2033-01-01','2034-12-31')]:
 q=z.loc[lo:hi].ic; print(label,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
for h in [5,10,20]:
 yy=p.pct_change(h).shift(-h); q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1]))
 print('decay',h,np.nanmean(q),len(q))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20340120_volume_confirmed_trend_signal.csv',index=False)
