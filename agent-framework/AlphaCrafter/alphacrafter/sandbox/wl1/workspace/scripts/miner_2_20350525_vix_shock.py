import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,5000)
 if x is None:x=get_index_daily_data(s,5000)
 if x is not None:D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
vx=get_index_daily_data('VIX',5000)
v=vx.assign(date=pd.to_datetime(vx.date)).set_index('date').close.astype(float).reindex(p.index).ffill()
# VIX shock reversal: short-term contrarian signal is stronger after a causal VIX jump.
# Shock is standardized 5d VIX change relative to trailing 120d level volatility, clipped.
vchg=v.pct_change(5)
scale=vchg.rolling(120).std().replace(0,np.nan)
shock=(vchg/scale).clip(-3,3)
for look in [3,5,10]:
 sig=(-r.rolling(look).sum().mul(1+shock.clip(lower=0),axis=0)).shift(1)
 for h in [5,10,20,40]:
  vals=[]; ds=[]; ns=[]
  for i,d in enumerate(sig.index):
   if i+h>=len(p): continue
   z=pd.concat([sig.loc[d],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].std()>0:
    vals.append(z.iloc[:,0].corr(z.iloc[:,1]));ds.append(d);ns.append(len(z))
  a=np.asarray(vals); print('look',look,'h',h,'dates',len(a),'avgN',np.mean(ns) if ns else 0,'coverage',np.mean(ns)/15 if ns else 0,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a),'hit',np.mean(a>0) if len(a) else 0)
 if look==3:
  print('regimes')
  for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
   q=a[(pd.DatetimeIndex(ds)>=pd.Timestamp(lo+'-01-01'))&(pd.DatetimeIndex(ds)<=pd.Timestamp(hi+'-12-31'))]
   print(lo, len(q), np.nanmean(q), np.nanmean(q)/np.nanstd(q) if len(q)>1 else np.nan)
  print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
  pd.DataFrame({'date':ds,'ic':a}).to_csv('scripts/miner_2_20350525_vix_shock_ic.csv',index=False)
  sig.to_csv('scripts/miner_2_20350525_vix_shock_signal.csv')
