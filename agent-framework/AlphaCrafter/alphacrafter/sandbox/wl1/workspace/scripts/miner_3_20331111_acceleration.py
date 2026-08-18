import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s,days=5000)
    except Exception: x=None
    if x is not None and len(x)>0:
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); f=(p.pct_change(20)-0.5*p.pct_change(60)/3.0).shift(1).replace([np.inf,-np.inf],np.nan)
fr=p.pct_change(10).shift(-10); rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mu=z.ic.mean(); sd=z.ic.std(); turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print('assets',len(D),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15); print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(mu,mu/sd,(z.ic>0).mean(),turn))
for label,lo,hi in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033','2033-01-01','2033-10-26')]:
 q=z.loc[lo:hi].ic; print(label,len(q),q.mean(),q.mean()/q.std())
print('decay')
for h in [5,10,20]:
 y=p.pct_change(h).shift(-h); q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1]))
 print(h,np.nanmean(q),np.nanmean(q)/np.nanstd(q))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20331111_acceleration_signal.csv',index=False)
