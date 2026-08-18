import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x)>100:return x[['date','close']]
  except Exception: pass
A=[]
for s in U:
 x=fetch(s)
 if x is None: continue
 x=x.set_index('date').sort_index(); r=x.close.pct_change()
 # Directional efficiency: net move divided by path length, with volatility normalization
 net=x.close.pct_change(20)
 path=r.abs().rolling(20).sum()
 vol=r.rolling(40).std()*np.sqrt(20)
 eff=(net/(path+1e-10))/(vol+1e-10)
 A.append(pd.DataFrame({s:eff,s+'_r':r}))
d=pd.concat(A,axis=1).sort_index(); F=d[[s for s in U if s in d]].shift(1)
R=d[[s+'_r' for s in U if s+'_r' in d]].rename(columns=lambda x:x[:-2])
h=10; FR=(1+R).rolling(h).apply(np.prod,raw=True).shift(-h+1)-1
ics=[]; ns=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],FR.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
q=pd.Series(ics); recent=q.tail(250)
print('assets',len([s for s in U if s in d]),'dates',len(q),'avg_n',np.mean(ns),'min_n',min(ns),'coverage',F.notna().sum(axis=1).mean()/15)
print('H10 IC %.6f ICIR %.6f hit %.4f recent250 IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),recent.mean(),recent.mean()/recent.std(ddof=1)))
for h2 in [1,5,10,20]:
 fr=(1+R).rolling(h2).apply(np.prod,raw=True).shift(-h2+1)-1; v=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h2,np.mean(v),len(v))
