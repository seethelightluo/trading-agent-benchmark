import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base=Path('../persistent/stock_data'); px={}
for s in U:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); px[s]=d
p=pd.DataFrame(px).sort_index().ffill(); cut=pd.Timestamp('2028-04-05'); r=p.pct_change(); fwd=p.shift(-10)/p-1
for w in [5,10,15,20,30]:
 fac=(p/p.shift(w)-1)*(.5+.5*r.gt(0).rolling(w).mean()); fac=fac.shift(1); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  if dt>cut: continue
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 x=pd.Series(vals,index=ds); print('w',w,'dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'early',x.iloc[:len(x)//2].mean(),'late',x.iloc[len(x)//2:].mean())
