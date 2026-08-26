import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 try:d=get_index_daily_data(s,4200)
 except Exception:
  try:d=get_stock_daily_data(s,4200)
  except Exception:continue
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in P.items()});hi=pd.DataFrame({s:d.high for s,d in P.items()});lo=pd.DataFrame({s:d.low for s,d in P.items()});ret=cl.pct_change();rv=ret.rolling(20).std(); loc=(2*cl-hi-lo)/(hi-lo).replace(0,np.nan)
# signed close location: close near high makes positive prior impulse, reversal is negative; near low makes positive reversal
sig=-(cl/cl.shift(5)-1)/(np.sqrt(5)*rv)*loc.rolling(3).mean()
rows=[]; fwd=cl.shift(-10)/cl.shift(-1)-1
for dt in cl.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):rows.append(q)
r=pd.Series(rows);print('dates',len(r),'avgN',15,'IC',r.mean(),'ICIR',r.mean()/r.std(),'hit',(r>0).mean(),'coverage',sig.notna().sum().sum()/sig.size)
for n in [252,756,1260]:
 q=r.tail(n);print('recent',n,q.mean(),q.mean()/q.std(),len(q))
for h in [5,10,20]:
 rr=[];fy=cl.shift(-h)/cl.shift(-1)-1
 for dt in cl.index:
  z=pd.concat([sig.loc[dt],fy.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):rr.append(q)
 print('decay',h,np.mean(rr),len(rr))
sig.to_csv('scripts/miner_2_20350806_close_location_signed_signal.csv')
