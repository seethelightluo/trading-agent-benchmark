import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
def eff(n): return px.pct_change(n)/(r.abs().rolling(n).sum()+1e-12)/(r.rolling(n).std()+1e-12)
# acceleration in directional path efficiency, robustly winsorized, lagged
f=eff(20)-eff(60)
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0).shift(1)
for h in [5,10,20]:
 y=px.pct_change(h).shift(-h); z=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if np.isfinite(c):z.append(c);ns.append(len(a))
 z=np.array(z); print('H',h,'dates',len(z),'avgN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(252),'hit',np.mean(z>0))
 if h==10:
  print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lo,hi in [(2020,2023),(2024,2026),(2027,2029),(2030,2032),(2033,2034)]:
 y=px.pct_change(10).shift(-10);z=[]
 for dt in f.index:
  if lo<=dt.year<=hi:
   a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
   if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 z=np.array(z);print('REG',lo,hi,'n',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1)*np.sqrt(252))
out=f.copy();out.insert(0,'date',out.index);out.to_csv('scripts/miner_2_20340217_efficiency_acceleration_signal.csv',index=False)
print('assets',len(P),'range',px.index.min(),px.index.max())
