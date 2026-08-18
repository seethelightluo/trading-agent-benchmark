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
ser={}
for s in U:
 x=fetch(s)
 if x is not None: ser[s]=x.set_index('date').sort_index().close
P=pd.concat(ser,axis=1).sort_index(); R=P.pct_change(); names=[s for s in U if s in P]
m=R[names].mean(axis=1); F=pd.DataFrame(index=P.index)
for s in names:
 beta=R[s].rolling(60).cov(m)/(m.rolling(60).var()+1e-12)
 F[s]=(R[s]-beta*m).rolling(20).sum().shift(1)
for h in [5,10,20]:
 fr=(1+R[names]).rolling(h).apply(np.prod,raw=True).shift(-h+1)-1; q=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=pd.Series(q); recent=q.tail(250)
 print('H %d dates %d avg_n %.2f min_n %d IC %.6f ICIR %.6f hit %.4f recent %.6f/%.6f'%(h,len(q),np.mean(ns),min(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),recent.mean(),recent.mean()/recent.std(ddof=1)))
print('coverage %.4f instruments %d dates %d'%(F.notna().sum(axis=1).mean()/15,len(names),len(F)))
