import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=4000) for s in U}
# Residual short-term reversal: remove common equal-weight market move, volatility-scale the residual.
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=np.log(px).diff(); m=r.mean(axis=1)
for w in [2,3,5,10]:
  resid=r.sub(m,axis=0); sig=-(resid.rolling(w,min_periods=w).sum()/ (r.rolling(20,min_periods=10).std()+1e-8)).shift(1)
  fwd=px.pct_change(1).shift(-1); out=[]
  for dt in sig.index:
   z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1: out.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
  q=pd.DataFrame(out,columns=['date','ic','n']); a=q.ic.to_numpy()
  print('W',w,'dates',len(a),'avgN',q.n.mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage',q.n.mean()/15)
  print('years',q.assign(year=q.date.dt.year).groupby('year').ic.mean().round(4).to_dict())
