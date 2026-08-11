import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-29'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date') for s in U}
def factor(x):
 gap=x.open/x.close.shift(1)-1; intr=x.close/x.open-1
 return -(gap+intr)/(x.close.pct_change().rolling(20,min_periods=10).std())
def run(h):
 rec=[]
 for s,x in D.items(): rec.append(pd.DataFrame({'date':x.index,'f':factor(x).values,'y':(x.close.shift(-h)/x.close-1).values}))
 a=pd.concat(rec,ignore_index=True).dropna(); vals=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1: vals.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
 return pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
z=run(1); valid=sum(factor(x).notna().sum() for x in D.values()); total=sum(len(x) for x in D.values())
print('overnight+intraday total reversal; cutoff',cut,'dates',len(z),'avg_n',z.n.mean(),'coverage',valid/total,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-29')]:
 q=z.loc[lo:hi].ic; print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [5,10]:
 q=run(h); print('decay',h,len(q),q.ic.mean())
r=pd.concat([factor(x).rename(s) for s,x in D.items()],axis=1).rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean(),'symbols',len(D))
