import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x)>100:return x[['date','close']]
  except:pass
A=[]
for s in U:
 x=g(s)
 if x is not None:
  x=x.set_index('date').sort_index(); r=x.close.pct_change()
  A.append(pd.DataFrame({s+'_r':r,s+'_m':x.close.pct_change(10),s+'_v':r.rolling(20).std()}))
d=pd.concat(A,axis=1).sort_index(); R=d[[s+'_r' for s in U]].rename(columns=lambda x:x[:-2]); M=d[[s+'_m' for s in U]].rename(columns=lambda x:x[:-2]); V=d[[s+'_v' for s in U]].rename(columns=lambda x:x[:-2])
# risk-adjusted medium-term momentum, smoothed and lagged
f=(M/(V*np.sqrt(10)+1e-8)).rolling(3,min_periods=3).mean().shift(1); fr=R.shift(-1)
q=[];ns=[]
for t in f.index:
 z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
 if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
q=pd.Series(q); rec=q.tail(250)
print('dates',len(q),'avg_n',np.mean(ns),'min_n',min(ns),'coverage',f.notna().sum(axis=1).mean()/15)
print('IC %.6f ICIR %.6f hit %.4f recent %.6f/%.6f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),rec.mean(),rec.mean()/rec.std(ddof=1)))
