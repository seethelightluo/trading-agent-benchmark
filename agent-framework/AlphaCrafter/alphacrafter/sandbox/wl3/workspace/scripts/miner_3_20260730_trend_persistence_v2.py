import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-29'; rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index(); r=x.close.pct_change(); f=r.rolling(20,min_periods=15).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-12); y=x.close.shift(-1)/x.close-1
 rows.append(pd.DataFrame({'date':x.index,'s':s,'f':f,'y':y}))
a=pd.concat(rows).dropna(); z=[]; ns=[]
for d,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1:
  c=spearmanr(g.f,g.y).statistic
  if pd.notna(c):z.append((d,c));ns.append(len(g))
z=pd.DataFrame(z,columns=['date','ic']).set_index('date');q=z.ic
print('candidate trend_persistence_20d_volnorm cutoff',cut,'dates',len(q),'avg_n',np.mean(ns),'coverage',len(a)/sum(len(pd.read_csv('../persistent/stock_data/'+s+'.csv')) for s in U),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank').diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-29')]:
 v=z.loc[lo:hi].ic;print('regime',lo,hi,len(v),v.mean(),v.mean()/v.std(ddof=1))
for h in [3,5,10]:
 vals=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index();r=x.close.pct_change();f=r.rolling(20,min_periods=15).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-12); y=x.close.shift(-h)/x.close-1; vals.append(pd.DataFrame({'date':x.index,'f':f,'y':y}))
 b=pd.concat(vals).dropna();vv=[]
 for d,g in b.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:vv.append(spearmanr(g.f,g.y).statistic)
 print('decay',h,len(vv),np.mean(vv),np.mean(vv)/np.std(vv,ddof=1))
