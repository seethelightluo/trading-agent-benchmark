import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=2600)
   if x is not None and len(x): return x
  except: pass
p={s:fetch(s) for s in U}; c=pd.concat({s:x.set_index('date')['close'] for s,x in p.items()},axis=1).sort_index(); r=np.log(c).diff()
# Residual medium reversal: remove same-day cross-asset common move from each asset's 5d return, then contrarian score.
ret5=np.log(c/c.shift(5)); common=ret5.median(axis=1); resid=ret5.sub(common,axis=0)
vol60=r.rolling(60).std()*np.sqrt(5)
raw=-resid/vol60
# smooth the residual shock with a secondary 2d component, maintaining interpretability
raw=raw + 0.25*(-resid.shift(2)/(r.rolling(60).std()*np.sqrt(5)))
sig=raw.sub(raw.median(axis=1),axis=0).shift(1); sig.to_csv('scripts/miner_3_20300321_residual_medium_reversal_signal.csv')
print('dates',len(c),'instruments',c.shape[1],'range',c.index.min(),c.index.max())
for h in [1,3,5,10]:
 f=np.log(c.shift(-h)/c); q=[]; nn=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:
   v=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(v):q.append(v);nn.append(len(z))
 q=pd.Series(q);print('h',h,'obs',len(q),'avg_n',np.mean(nn),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('coverage',sig.notna().sum(axis=1).mean()/len(U),'turnover',sig.rank(pct=True).diff().abs().mean().mean())
