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
P=pd.concat(ser,axis=1).sort_index(); R=P.pct_change(); names=list(P.columns)
ret20=P.pct_change(20); consistency=(R.gt(0).rolling(20).mean()-0.5)*2
F=(ret20*consistency).shift(1)
for h in [1,5,10,20]:
 fr=P.pct_change(h).shift(-h); vals=[]; dates=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 q=pd.Series(vals,index=pd.DatetimeIndex(dates)); recent=q.tail(250); online=q[q.index>=pd.Timestamp('2026-07-16')]
 print('H %d dates %d avg_n %.2f min_n %d IC %.6f ICIR %.6f hit %.4f recent %.6f/%.6f online %.6f/%.6f'%(h,len(q),np.mean(ns),min(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),recent.mean(),recent.mean()/recent.std(ddof=1),online.mean(),online.mean()/online.std(ddof=1)))
rank=F.rank(axis=1,pct=True); print('coverage %.4f instruments %d rows %d turnover %.6f'%(F.notna().sum().sum()/(F.shape[0]*15),len(names),len(F),rank.diff().abs().mean(axis=1).mean()))
