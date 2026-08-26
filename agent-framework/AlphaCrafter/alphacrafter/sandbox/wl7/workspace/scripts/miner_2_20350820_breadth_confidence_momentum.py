import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=None
 for f in (get_index_daily_data,get_stock_daily_data):
  try:d=f(s,4200)
  except Exception:d=None
  if d is not None:break
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in P.items()}); ret=cl.pct_change(); vol=ret.rolling(40,min_periods=20).std()
# medium horizon trend, normalized by volatility and conditioned on breadth; lag all inputs
mom=(cl/cl.shift(20)-1)/(vol*np.sqrt(20)); breadth=(ret.rolling(20).sum()>0).mean(axis=1).shift(1)
# continuous breadth confidence: market-wide agreement away from 50%, preserves sign of trend
sig=mom.shift(1).mul((breadth-0.5)*2,axis=0)
sig=sig.sub(sig.median(axis=1),axis=0)
rows=[]
for dt in sig.index:
 y=cl.shift(-10).loc[dt]/cl.shift(-1).loc[dt]-1;z=pd.concat([sig.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):rows.append((dt,q,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');x=a.ic
print('assets',len(cl.columns),'dates',len(a),'avgN',a.n.mean(),'coverage',sig.notna().mean().mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for n in [252,756,1260]:q=x.tail(n);print('recent',n,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [5,10,20]:
 y=cl.shift(-h)/cl.shift(-1)-1;rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 rr=pd.Series(rr).dropna();print('decay',h,len(rr),rr.mean(),rr.mean()/rr.std(ddof=1))
sig.to_csv('scripts/miner_2_20350820_breadth_confidence_momentum_signal.csv')
