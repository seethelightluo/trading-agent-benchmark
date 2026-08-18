import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None: D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std(); imp=(v20/(v60+1e-8)-1).clip(lower=0,upper=3)
# Volatility impulse continuation: trend is trusted only when volatility is freshly expanding.
f=(r.rolling(5).sum()/(v20+1e-8)*imp).shift(1); y=p.pct_change(10).shift(-10); rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mu=z.ic.mean(); sd=z.ic.std(); print('assets',len(D),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15); print('IC',mu,'ICIR',mu/sd,'hit',(z.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lab,lo,hi in [('2020-23','2020','2023'),('2024-26','2024','2026'),('2027-29','2027','2029'),('2030-32','2030','2032'),('2033-34','2033','2034')]:
 q=z.loc[lo:hi].ic; print(lab,len(q),q.mean(),q.mean()/q.std())
for h in [1,5,10,20]:
 yy=p.pct_change(h).shift(-h); q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1]))
 print('decay',h,np.nanmean(q),np.nanmean(q)/np.nanstd(q))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_3_20340120_vol_impulse_continuation_signal.csv',index=False)
