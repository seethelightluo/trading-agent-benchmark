import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index()
vol=pd.DataFrame({s:d.set_index('date')['volume'] for s,d in D.items() if d is not None}).reindex(px.index)
ret=px.pct_change(20); vr=np.log1p(vol).rolling(20).mean()-np.log1p(vol).rolling(60).mean(); f=(ret*vr).shift(1)
for h in [5,10,20,40]:
 fr=px.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(dt)
 a=np.array(vals); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),5),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),5),'hit',round(np.mean(a>0),4))
 for lo,hi in [('2020','2024'),('2025','2029'),('2030','2035')]:
  q=a[(np.array(dates)>=pd.Timestamp(lo+'-01-01'))&(np.array(dates)<=pd.Timestamp(hi+'-12-31'))];print(lo,round(np.nanmean(q),5),round(np.nanmean(q)/np.nanstd(q,ddof=1),4),len(q))
print('coverage',round(f.notna().mean().mean(),4))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20350216_volume_confirmed_momentum_signal.csv',index=False)
