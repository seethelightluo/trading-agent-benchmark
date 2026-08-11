import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-09-09'
def frame(h):
 rows=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date')
  r=x.close.pct_change(); vol=r.rolling(20,min_periods=10).std()
  f=-r.rolling(3,min_periods=3).sum()/(vol*np.sqrt(3)+1e-12)
  y=x.close.shift(-h)/x.close-1
  rows.append(pd.DataFrame({'date':x.date,'symbol':s,'f':f,'y':y}))
 return pd.concat(rows,ignore_index=True).dropna()
def calc(a):
 out=[]; ns=[]
 for d,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:
   c=spearmanr(g.f,g.y).statistic
   if pd.notna(c):out.append((d,c));ns.append(len(g))
 return pd.DataFrame(out,columns=['date','ic']).set_index('date'),ns
a=frame(1); z,ns=calc(a); q=z.ic
rank=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank')
print('candidate volscaled_reversal_3d cutoff',cut,'dates',len(q),'avg_n',np.mean(ns),'coverage',len(a)/(sum(len(pd.read_csv('../persistent/stock_data/'+s+'.csv')) for s in U)),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',rank.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-09-09')]:
 v=z.loc[lo:hi].ic; print('regime',lo,hi,'n',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1) if len(v)>1 else np.nan)
for h in [3,5,10]:
 oo,_=calc(frame(h)); print('decay',h,'n',len(oo),'IC',oo.ic.mean(),'ICIR',oo.ic.mean()/oo.ic.std(ddof=1))
a[['date','symbol','f']].rename(columns={'f':'signal'}).to_csv('scripts/miner_3_20260910_volscaled_reversal_3d_signal.csv',index=False)
