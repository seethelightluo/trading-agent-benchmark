import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
D={}
for s in U:
 d=get_stock_daily_data(symbol=s,days=4000)
 if d is not None:D[s]=d.sort_values('date').set_index('date')['close'].astype(float)
ix=sorted(set.intersection(*[set(x.index) for x in D.values()])); P=pd.DataFrame({s:D[s].reindex(ix) for s in U},index=ix).ffill(); R=P.pct_change(); vol=R.rolling(20).std(); r5=P.pct_change(5); breadth=(P.pct_change(20)>0).mean(1); fwd=P.shift(-1)/P-1
for th in [.30,.35,.40,.45,.50,.55,.60]:
 sig=(-r5/vol).where(breadth<th).shift(1); a=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print('th',th,'n',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'avgN',np.mean(ns),'frac',np.isfinite(sig).mean().mean())
