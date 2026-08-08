import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Trend consistency: 20d signed breadth times 20d return, normalized by 20d vol.
px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); px[s]=z.drop_duplicates('date').set_index('date').close
c=pd.DataFrame(px).sort_index(); r=c.pct_change()
# interpretable factor: return * (2*positive-day fraction-1), with volatility scaling
breadth=(r>0).rolling(20,min_periods=15).mean()*2-1
vol=r.rolling(20,min_periods=15).std()
f=(c.pct_change(20)*breadth/vol).replace([np.inf,-np.inf],np.nan)
# forward non-overlapping-style horizons
for h in [1,5,10]:
 fw=c.pct_change(h).shift(-h); vals=[]
 for dt in f.index:
  q=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(q)>=8 and q.x.nunique()>1 and q.y.nunique()>1: vals.append(q.x.corr(q.y))
 by=pd.Series(vals).dropna(); ic=by.mean(); sd=by.std(ddof=1)
 print('horizon',h,'dates',len(by),'obs',int(fw.notna().sum().sum()),'avgN',round(len(f.stack())/len(f.index),2),'IC %.6f ICIR %.6f hit %.4f'%(ic,ic/sd*np.sqrt(252), (by>0).mean()))
rank=f.rank(axis=1,pct=True)
print('coverage',f.notna().sum().sum()/(len(f.index)*len(U)),'turnover',rank.diff().abs().mean(axis=1).mean())
print('period',f.index.min(),f.index.max())
