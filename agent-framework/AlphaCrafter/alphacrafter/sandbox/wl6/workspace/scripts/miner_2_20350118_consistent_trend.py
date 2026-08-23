import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=6000)
 if d is not None and len(d):
  x=d.copy(); x.date=pd.to_datetime(x.date); px[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index()
# 60d trend weighted by consistency of daily direction, lagged one completed day
r=P.pct_change()
trend=P/P.shift(60)-1
cons=(r>0).rolling(60,min_periods=45).mean()
f=(trend*cons).shift(1)
print('universe',len(px),'dates',P.index.min(),P.index.max())
for h in [5,10,20,40]:
 fr=P.shift(-h)/P-1
 vals=[]; ns=[]; turns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 # turnover based rank changes, daily valid dates
 for i in range(1,len(f)):
  a=f.iloc[i-1].rank(pct=True); b=f.iloc[i].rank(pct=True)
  q=pd.concat([a,b],axis=1).dropna()
  if len(q)>=8: turns.append((q.iloc[:,0]-q.iloc[:,1]).abs().mean())
 v=np.array(vals); print('H',h,'dates',len(v),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',np.nanmean(v),'ICIR',np.nanmean(v)/np.nanstd(v,ddof=1),'hit',np.mean(v>0),'turn',np.nanmean(turns))
 # regimes for 40
 if h==40:
  dts=np.array([d for d in f.index if d in fr.index])
  for a,b in [('2020','2027-12-31'),('2028','2031-12-31'),('2032','2035-01-18')]:
   mask=[str(x)>=a and str(x)<=b for x in dts[:len(v)]]
   print('regime',a,b,np.nanmean(v[mask]) if any(mask) else np.nan, sum(mask))
# save artifact latest factor signal, enough provenance
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20350118_consistent_trend_signal.csv',index=False)
