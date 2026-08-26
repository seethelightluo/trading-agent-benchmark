import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,6000).set_index('date')['close'].astype(float).rename(s) for s in U}
P=pd.concat(D.values(),axis=1).sort_index().ffill(); R=P.pct_change()
# Volatility carry: prefer assets with low recent realized volatility, but use
# a causal volatility trend adjustment so falling volatility receives a premium.
v20=R.rolling(20,min_periods=15).std(); v60=R.rolling(60,min_periods=40).std()
f=(-np.log(v20+1e-12) + 0.5*np.log((v60+1e-12)/(v20+1e-12))).replace([np.inf,-np.inf],np.nan)
for h in [5,10,20]:
 fr=P.shift(-h).div(P)-1; vals=[]; ds=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c);ds.append(dt);ns.append(len(z))
 a=np.array(vals); ds=pd.DatetimeIndex(ds)
 print('horizon',h,'dates',len(a),'meanN',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),6))
 if h==10:
  for aa,bb in [('2023-11-13','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-10-10')]:
   q=a[(ds>=aa)&(ds<=bb)];print('regime',aa,bb,len(q),round(q.mean(),6) if len(q) else None)
  print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
out=f.loc['2020':].stack().rename('factor_value').reset_index();out.columns=['date','symbol','factor_value'];out.to_csv('scripts/miner_2_20351011_vol_carry_signal.csv',index=False)
