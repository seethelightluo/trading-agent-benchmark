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
 if d is not None and len(d)>100:d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in P.items()});r=cl.pct_change();v=r.rolling(40,min_periods=20).std();
# blend 20/60 trend, volatility normalized, with broad-market direction confidence
m=(.6*(cl/cl.shift(20)-1)/ (v*np.sqrt(20)) + .4*(cl/cl.shift(60)-1)/(v*np.sqrt(60))).shift(1)
b=(r.rolling(20).sum()>0).mean(axis=1).shift(1);sig=m.mul((b-.5)*2,axis=0).sub(m.mul((b-.5)*2,axis=0).median(axis=1),axis=0)
rows=[]
for dt in sig.index:
 y=cl.shift(-10).loc[dt]/cl.shift(-1).loc[dt]-1;z=pd.concat([sig.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):rows.append(q)
x=pd.Series(rows);print('assets',len(cl.columns),'dates',len(x),'coverage',sig.notna().mean().mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for n in [252,756,1260]:q=x.tail(n);print('recent',n,q.mean(),q.mean()/q.std(ddof=1))
sig.to_csv('scripts/miner_2_20350820_multihorizon_breadth_signal.csv')
