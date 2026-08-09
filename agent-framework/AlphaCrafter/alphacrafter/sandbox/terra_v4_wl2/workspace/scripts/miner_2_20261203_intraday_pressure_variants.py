import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); intr=d.close/d.open-1; rng=(d.high-d.low)/d.close
 for n in [2,3,4]:
  f=-(intr/rng.replace(0,np.nan)).rolling(n,min_periods=n).mean(); y=d.close.pct_change().shift(-1)
  rows.append(pd.DataFrame({'date':d.date,'s':s,'f':f,'y':y,'n':n}))
a=pd.concat(rows)
for n,g0 in a.groupby('n'):
 out=[]
 for dt,g in g0.dropna(subset=['f','y']).groupby('date'):
  if len(g)>=8: out.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
 v=np.array([x[1] for x in out]); print(n,len(v),np.mean([x[2] for x in out]),v.mean(),v.mean()/v.std(ddof=1),(v>0).mean())
