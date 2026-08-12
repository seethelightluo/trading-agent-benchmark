import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2026-07-15'
def load(p): return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().query('date<=@CUT')
D={s:load('../persistent/stock_data/'+s+'.csv') for s in U}
C=pd.concat([x.close.pct_change().rename(s) for s,x in D.items()],axis=1,sort=True); bench=C.mean(axis=1)
def factor(x,s):
 r=x.close.pct_change(); resid=r-bench.reindex(x.index)
 return (resid.rolling(10,min_periods=8).sum()/(resid.rolling(20,min_periods=15).std()*np.sqrt(10))).rename(s)
rows=[]
for s,x in D.items(): rows.append(pd.DataFrame({'date':x.index.to_numpy(),'f':factor(x,s).to_numpy(),'y':(x.close.shift(-1)/x.close-1).to_numpy(),'s':s}))
a=pd.concat(rows,ignore_index=True).dropna(); out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: out.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
alln=sum(len(x) for x in D.values()); valid=sum(factor(x,s).notna().sum() for s,x in D.items())
ranks=pd.concat([factor(x,s) for s,x in D.items()],axis=1).rank(axis=1,pct=True)
print('idea relative 10d trend volscaled; cutoff',CUT,'dates',len(z),'avg_n',z.n.mean(),'coverage',valid/alln,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean(),'turnover',ranks.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=z.loc[lo:hi].ic; print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [3,5,10]:
 b=pd.concat([pd.DataFrame({'date':x.index.to_numpy(),'f':factor(x,s).to_numpy(),'y':(x.close.shift(-h)/x.close-1).to_numpy()}) for s,x in D.items()],ignore_index=True).dropna(); q=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
 q=pd.Series(q); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
