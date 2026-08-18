import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); P[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); ret=px.pct_change(20); path=r.abs().rolling(20).sum(); dn=r.clip(upper=0).rolling(20).std()
# Directional path efficiency, rewarded for persistent upside and penalized by downside volatility.
f=(ret/(path+1e-12))/(dn+1e-12); f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0).shift(1)
for h in [5,10,20]:
 y=px.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c); ns.append(len(a)); dates.append(dt)
 z=np.array(vals); print('H',h,'dates',len(z),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC %.8f dailyICIR %.8f annualICIR %.5f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),z.mean()/z.std(ddof=1)*np.sqrt(252),np.mean(z>0)))
 if h==20:
  f.loc[dates].stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20340217_downside_efficiency_20d_signal.csv',index=False)
 if h==10:
  print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
  f.loc[dates].stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20340217_downside_efficiency_signal.csv',index=False)
for start,end in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 y=px.pct_change(10).shift(-10); z=[]
 for dt in f.index:
  if not(start<=str(dt.year)<=end): continue
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 z=np.array(z); print('REG',start,end,'n',len(z),'IC %.6f dailyICIR %.6f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1)))
print('assets',len(P),'range',px.index.min(),px.index.max())
