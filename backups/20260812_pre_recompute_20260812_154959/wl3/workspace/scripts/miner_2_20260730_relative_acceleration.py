import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date') for s in U}
# Volatility-scaled relative acceleration: recent 5d return versus expected 5d slice of 20d trend.
def fac(x):
 r=x.close.pct_change(); accel=x.close.pct_change(5)-x.close.pct_change(20)/4
 return accel/(r.rolling(20).std()*np.sqrt(5)+1e-12)
def run(h):
 z=[]
 for s,x in D.items(): z.append(pd.DataFrame({'date':x.index,'f':fac(x).values,'y':(x.close.shift(-h)/x.close-1).values,'s':s}))
 a=pd.concat(z,ignore_index=True).dropna(); out=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: out.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
z=run(1); valid=sum(fac(x).notna().sum() for x in D.values()); total=sum(len(x) for x in D.values())
print('relative acceleration volscaled; cutoff',cut,'dates',len(z),'avg_n',z.n.mean(),'coverage',valid/total,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=z.loc[lo:hi].ic; print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [5,10,20]:
 q=run(h); print('decay',h,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
r=pd.concat([fac(x).rename(s) for s,x in D.items()],axis=1).rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean(),'symbols',len(D))
