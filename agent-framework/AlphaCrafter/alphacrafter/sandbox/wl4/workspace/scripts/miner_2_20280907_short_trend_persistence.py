import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x)>100:return x[['date','close']]
  except:pass
D=[]
for s in U:
 x=get(s)
 if x is not None:
  x=x.set_index('date').sort_index(); r=x.close.pct_change();
  # recent trend, scaled by medium volatility; one-day lag
  f=x.close.pct_change(5)/(r.rolling(20).std()*np.sqrt(5)+1e-9)
  D.append(pd.DataFrame({s:f,s+'_r':r}))
d=pd.concat(D,axis=1).sort_index(); F=d[[s for s in U if s in d]].shift(1); R=d[[s+'_r' for s in U if s+'_r' in d]].rename(columns=lambda z:z[:-2])
for h in [5,10,20]:
 fr=(1+R).rolling(h).apply(np.prod,raw=True).shift(-h+1)-1; q=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=pd.Series(q); recent=q.tail(250)
 print('H',h,'dates',len(q),'avg_n',np.mean(ns),'min_n',min(ns),'IC %.6f ICIR %.6f hit %.4f recent %.6f/%.6f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),recent.mean(),recent.mean()/recent.std(ddof=1)))
print('coverage',F.notna().sum(axis=1).mean()/15)
