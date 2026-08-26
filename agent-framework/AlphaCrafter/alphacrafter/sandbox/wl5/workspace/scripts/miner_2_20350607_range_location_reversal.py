import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=180: frames[s]=d.set_index('date')
# Range-location reversal: fade medium-term residual move, weighted toward assets
# whose recent closes sit near the lower/upper edge of their daily ranges.
C=pd.DataFrame({s:d.close.astype(float) for s,d in frames.items()}).sort_index()
R=C.pct_change(); resid=R.sub(R.mean(axis=1),axis=0)
rev=-resid.rolling(20,min_periods=15).sum()/(resid.rolling(60,min_periods=40).std()*np.sqrt(20)+1e-12)
high=pd.DataFrame({s:d.high.astype(float) for s,d in frames.items()}).reindex(C.index)
low=pd.DataFrame({s:d.low.astype(float) for s,d in frames.items()}).reindex(C.index)
# persistent close location: positive means closes near high; reversal signal fades it
loc=((C-low)/(high-low+1e-12)-0.5).rolling(10,min_periods=5).mean()
sig=(rev*(1-0.8*loc)).clip(-8,8)
print('assets',len(C.columns),'rows',len(C),'dates',C.index.min().date(),C.index.max().date())
for h in [5,10,20]:
 fwd=C.shift(-h)/C-1; vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(x): vals.append(x); dates.append(dt); ns.append(len(z))
 a=np.array(vals); dates=pd.DatetimeIndex(dates); ns=np.array(ns)
 print('horizon',h,'dates',len(a),'mean_n',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),6))
 if h==10 and len(a):
  for x,y in [('2024-01-01','2025-12-31'),('2026-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2035-06-07')]:
   w=a[(dates>=x)&(dates<=y)]; print('regime',x,y,'dates',len(w),'IC',round(w.mean(),6) if len(w) else None)
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in dates],index=dates)
  print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20350607_range_location_reversal_signal.csv',index=False)
