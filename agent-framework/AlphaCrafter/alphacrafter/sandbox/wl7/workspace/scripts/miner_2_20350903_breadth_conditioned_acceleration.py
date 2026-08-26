import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=None
 for f in (get_index_daily_data,get_stock_daily_data):
  try: d=f(s,4200)
  except Exception: d=None
  if d is not None: break
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in P.items()}); r=cl.pct_change(); vol=r.rolling(40,min_periods=20).std()
# Novel: acceleration (short trend minus long trend), normalized by volatility,
# activated only when breadth agrees with the sign of each asset's long trend.
t20=cl/cl.shift(20)-1; t60=cl/cl.shift(60)-1
acc=(t20-t60/3)/(vol*np.sqrt(20))
breadth=(t20>0).mean(axis=1).shift(1)
confidence=(2*breadth-1).abs()
long_sign=np.sign(t60).replace(0,np.nan)
sig=(acc*long_sign).mul(confidence,axis=0).shift(1)
rows=[]; dates=[]; counts=[]
for dt in sig.index:
 y=cl.shift(-10).loc[dt]/cl.shift(-1).loc[dt]-1
 z=pd.concat([sig.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): rows.append(q); dates.append(dt); counts.append(len(z))
x=pd.Series(rows,index=pd.to_datetime(dates));
print('assets',len(cl.columns),'dates',len(x),'avg_n',np.mean(counts),'coverage',sig.notna().mean().mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for n in [252,756,1260]:
 q=x.tail(n); print('recent',n,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
sig.to_csv('scripts/miner_2_20350903_breadth_conditioned_acceleration_signal.csv')
