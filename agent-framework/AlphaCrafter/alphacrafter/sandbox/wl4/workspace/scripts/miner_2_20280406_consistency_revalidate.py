import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); px={}
for s in U:
 d=pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index(); px[s]=d
p=pd.DataFrame(px).sort_index().ffill(); cut=pd.Timestamp('2028-04-05')
r=p.pct_change(); mom=p/p.shift(15)-1; consistency=(r.gt(0).rolling(15).mean()); fac=(mom*(.5+.5*consistency)).shift(1)
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  if dt>cut: continue
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 x=pd.Series(vals,index=ds); print('H',h,'dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0))
 for label,lo,hi in [('early','2020','2024-01-01'),('late','2024-01-01','2028-04-06')]:
  q=x.loc[lo:hi]; print(label,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
rank=fac.rank(axis=1,pct=True); print('coverage',fac.loc[:cut].notna().mean().mean(),'turnover',rank.diff().abs().mean(axis=1).loc[:cut].mean())
