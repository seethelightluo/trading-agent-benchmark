import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); disp=r.rolling(20).std().mean(axis=1); hi=(disp>disp.rolling(120).median()).astype(float)
# pullback within established trends, scaled by idiosyncratic recent volatility; lagged one day
trend=p.pct_change(60); f=(-p.pct_change(10)).div(r.rolling(20).std()+1e-8).mul((trend>0).astype(float),axis=0).mul(hi,axis=0).shift(1); y=p.pct_change(10).shift(-10)
rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mu=z.ic.mean(); sd=z.ic.std(); print('assets',len(D),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15); print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(mu,mu/sd,(z.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for label,lo,hi2 in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033','2033-01-01','2033-12-09')]:
 q=z.loc[lo:hi2].ic; print(label,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
for h in [5,10,20]:
 yy=p.pct_change(h).shift(-h); q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1]))
 print('decay',h,np.nanmean(q),np.nanmean(q)/np.nanstd(q))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20331209_trend_pullback_signal.csv',index=False)
