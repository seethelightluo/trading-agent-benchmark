import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date') for s in U}
# Candidate: volume-confirmed short-term reversal.  Prior-day return is reversed,
# scaled by 20d volatility and amplified when prior volume is unusually high.
def fac(x):
 r=x.close.pct_change(); vol=r.rolling(20,min_periods=10).std()
 if 'volume' in x: vratio=x.volume/x.volume.rolling(20,min_periods=10).median()
 else: vratio=pd.Series(1.,index=x.index)
 boost=(vratio.clip(0.5,3.0)-1).clip(lower=0)
 return -r/vol*(1+0.35*boost)
def run(h):
 rows=[]
 for s,x in D.items(): rows.append(pd.DataFrame({'date':x.index,'f':fac(x).values,'y':(x.close.shift(-h)/x.close-1).values}))
 a=pd.concat(rows,ignore_index=True).dropna(); out=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1: out.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
for h in [1,5,10]:
 z=run(h); print('h',h,'dates',len(z),'avg_n',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-29')]:
   q=z.loc[lo:hi].ic; print('regime',lo,hi,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
r=pd.concat([fac(x).rename(s) for s,x in D.items()],axis=1).rank(axis=1,pct=True); print('coverage',sum(fac(x).notna().sum() for x in D.values())/sum(len(x) for x in D.values()),'turnover',r.diff().abs().mean(axis=1).mean(),'symbols',len(D))
