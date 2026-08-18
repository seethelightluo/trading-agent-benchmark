import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d.sort_values('date').set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); mom=np.log(px/px.shift(60)); vol=r.rolling(40).std()*np.sqrt(40); sv=r.rolling(10).std()*np.sqrt(10)
f=(mom/(vol+1e-9))*(1/(1+sv/(vol+1e-9))).shift(1); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],np.log(px.shift(-20)/px).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('assets',len(raw),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15); print('IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'hit',(z.ic>0).mean(),'turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20]:
 q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],np.log(px.shift(-h)/px).loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(q),np.nanmean(q)/np.nanstd(q))
for label,lo,hi in [('2024-26','2024','2026'),('2027-29','2027','2029'),('2030-32','2030','2032'),('2033','2033','2033')]:
 q=z.loc[lo:hi].ic; print(label,len(q),q.mean(),q.mean()/q.std())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20331111_vol_regime_signal.csv',index=False)
