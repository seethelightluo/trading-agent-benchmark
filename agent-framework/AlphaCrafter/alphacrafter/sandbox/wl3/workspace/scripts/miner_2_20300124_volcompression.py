import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try:
   x=f(s,days=2600)
   if x is not None and len(x): return x
  except: pass
p={s:g(s) for s in U}; c=pd.concat({s:x.set_index('date')['close'] for s,x in p.items()},axis=1).sort_index(); r=np.log(c).diff()
# volatility contraction signal: prefer assets with contracting short vol, but rank by recent return
v10=r.rolling(10).std(); v60=r.rolling(60).std(); ret10=np.log(c/c.shift(10));
# compression + positive drift, demean cross section, lag
raw=(ret10/(v20:=r.rolling(20).std()) - 0.5*(v10/v60-1))
sig=raw.sub(raw.median(axis=1),axis=0).shift(1); sig.to_csv('scripts/miner_2_20300124_volcompression_signal.csv')
print('dates',len(c),'instruments',c.shape[1])
for h in [1,3,5,10]:
 f=np.log(c.shift(-h)/c); q=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print(h,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
print('coverage',sig.notna().sum(axis=1).mean()/15,'turnover',sig.rank(pct=True).diff().abs().mean().mean())
