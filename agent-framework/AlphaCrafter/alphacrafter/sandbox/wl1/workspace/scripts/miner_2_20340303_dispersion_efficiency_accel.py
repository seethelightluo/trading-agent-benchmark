import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
# Candidate: acceleration in directional range efficiency, conditioned by market dispersion.
e20=(px.pct_change(20)/(r.abs().rolling(20,min_periods=15).sum()+1e-12))/(r.rolling(20,min_periods=15).std()+1e-12)
e60=(px.pct_change(60)/(r.abs().rolling(60,min_periods=40).sum()+1e-12))/(r.rolling(60,min_periods=40).std()+1e-12)
acc=e20-e60
# dispersion regime is observable cross-sectional absolute return dispersion, lag signal by one day.
disp=r.rolling(10,min_periods=8).std().mean(axis=1)
drank=disp.rank(pct=True)
# High dispersion emphasizes acceleration; low dispersion mildly suppresses it.
f=(acc*(0.65+0.70*drank.values[:,None])).clip(acc.quantile(.05,axis=1),acc.quantile(.95,axis=1),axis=0).shift(1)
fr=px.pct_change(10).shift(-10)
rows=[]
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h); z=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if np.isfinite(c):z.append(c);ns.append(len(a))
 z=np.array(z); print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC %.8f ICIR %.8f hit %.4f'%(np.mean(z),np.mean(z)/np.std(z,ddof=1)*np.sqrt(252),np.mean(z>0)))
 if h==10:
  sig=f.rank(axis=1,pct=True); print('coverage %.6f turnover %.6f'%(f.notna().sum(axis=1).mean()/len(U),sig.diff().abs().mean(axis=1).dropna().mean()))
  out=f.loc[f.index[f.index.isin(f.index)]];out.insert(0,'date',out.index);out.to_csv('scripts/miner_2_20340303_dispersion_efficiency_accel_signal.csv',index=False)
for start,end in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 z=[]
 for dt in f.index:
  if not(start<=str(dt.year)<=end):continue
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 z=np.array(z);print('REG',start,end,'n',len(z),'IC %.8f'%(np.nanmean(z)))
print('range',px.index.min(),px.index.max(),'assets',len(P))
