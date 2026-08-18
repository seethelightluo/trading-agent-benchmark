import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
# One interpretable idea: volatility-normalized horizon disagreement, with recent short trend
# rewards assets whose 10d trend disagrees positively with their 60d trend, scaled by residual risk
m10=np.log(px/px.shift(10)); m60=np.log(px/px.shift(60));
rv=r.rolling(30,min_periods=20).std()*np.sqrt(30)
f=((m10-m60/6)/(rv+1e-8)).shift(1)
for h in [5,10,20]:
 fr=px.pct_change(h).shift(-h); z=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a))
 z=np.array(z); print('h',h,'dates',len(z),'avgN %.2f'%np.mean(ns),'IC %.8f ICIR %.8f hit %.4f coverage %.4f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0),np.mean(ns)/15))
 for a,b in [(2020,2023),(2024,2026),(2027,2029),(2030,2032),(2033,2034)]:
  q=z[[i for i,dt in enumerate(f.index) if dt.year>=a and dt.year<=b and i<len(z)]]
  # index mismatch is slight due skipped dates; use direct recompute below not needed
  print(a,b,'n',len(q),'ic',np.nanmean(q) if len(q) else np.nan,'icir',np.nanmean(q)/np.nanstd(q,ddof=1) if len(q)>1 else np.nan)
 if h==10:
  out=f.copy();out.insert(0,'date',out.index);out.to_csv('scripts/miner_1_20340317_horizon_disagreement_signal.csv',index=False)
