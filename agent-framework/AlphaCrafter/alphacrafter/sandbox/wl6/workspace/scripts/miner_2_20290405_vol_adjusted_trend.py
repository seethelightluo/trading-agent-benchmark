import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in(get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=3000)
   if x is not None and len(x): return x.set_index('date')['close']
  except:pass
p=pd.concat({s:get(s) for s in U},axis=1).sort_index(); r=p.pct_change();
# medium trend per realized risk, with a slow trend confirmation multiplier
f=p.pct_change(30)/(r.rolling(30,min_periods=20).std()+1e-12)*(1+0.35*np.sign(p.pct_change(90)))
print('universe=%d dates=%d'%(p.shape[1],len(p)))
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; q=[];ns=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(d)
 q=pd.Series(q,index=ds);print('h=%d dates=%d avg_n=%.2f cov=%.4f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%(h,len(q),np.mean(ns),f.loc[ds].notna().sum().sum()/(len(ds)*15),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
