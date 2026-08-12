import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=3000)
   if d is not None and len(d)>=100:
    d=d.copy();d.date=pd.to_datetime(d.date);return d.set_index('date').close.astype(float)
  except Exception:pass
p={s:g(s) for s in U};p={s:x for s,x in p.items() if x is not None};P=pd.concat(p,axis=1).sort_index().ffill();R=P.pct_change()
# Short trend normalized by realized volatility, with sign chosen for robust one-day cross-sectional continuation.
raw=P.pct_change(15)/(R.rolling(25,min_periods=18).std()*np.sqrt(252)+1e-8)
f=(-raw).sub((-raw).median(axis=1),axis=0).clip(-6,6)
for h in [1,3,5,10]:
 fr=P.shift(-h)/P-1;v=[];ds=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1]));ds.append(dt);ns.append(len(z))
 ic=pd.Series(v,index=ds).dropna();print('H',h,'obs',len(ic),'avgN',np.mean(ns),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1)*np.sqrt(len(ic)),'hit',(ic>0).mean())
 if h==1:print('coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for lab,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01','2027-08-11')]:
 x=ic[(ic.index>=a)&(ic.index<=b)];print('REG',lab,len(x),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(len(x)))
out=f.iloc[-1].rename('signal').to_frame();out.index.name='symbol';out.to_csv('scripts/miner_2_20270812_short_trend_contra_signal.csv');print('cutoff',f.dropna(how='all').index.max(),'dates',len(P),'instruments',len(P.columns))
