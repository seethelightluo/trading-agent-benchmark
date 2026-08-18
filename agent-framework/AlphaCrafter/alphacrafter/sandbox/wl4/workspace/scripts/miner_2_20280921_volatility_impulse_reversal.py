import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x)>100:return x[['date','close']]
  except Exception: pass
cols=[]
for s in U:
 x=fetch(s)
 if x is None: continue
 x=x.set_index('date').sort_index(); r=x.close.pct_change()
 # Volatility impulse: recent 5d realized vol relative to 60d baseline; high impulses tend to mean-revert.
 f=(r.rolling(5).std()/ (r.rolling(60).std()+1e-12)).shift(1)
 cols += [f.rename(s),r.rename(s+'_r')]
d=pd.concat(cols,axis=1).sort_index(); names=[s for s in U if s in d]
F=d[names]; R=d[[s+'_r' for s in names]].rename(columns=lambda x:x[:-2])
for h in [5,10,20]:
 fr=(1+R).rolling(h).apply(np.prod,raw=True).shift(-h+1)-1
 q=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=pd.Series(q); recent=q.tail(250)
 print('H',h,'dates',len(q),'avg_n %.2f min_n %d IC %.6f ICIR %.6f hit %.4f recent %.6f/%.6f'%(np.mean(ns),min(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),recent.mean(),recent.mean()/recent.std(ddof=1)))
print('coverage %.4f instruments %d dates %d'%(F.notna().sum(axis=1).mean()/15,len(names),len(F)))
