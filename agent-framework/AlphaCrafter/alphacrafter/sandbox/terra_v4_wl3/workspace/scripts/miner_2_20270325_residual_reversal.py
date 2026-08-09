import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2027-03-25')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date'); P[s]=d.close.replace(0,np.nan)
P=pd.DataFrame(P); r=P.pct_change(); m=r.mean(axis=1); f=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
 beta=r[s].rolling(60,min_periods=40).cov(m)/m.rolling(60,min_periods=40).var().replace(0,np.nan)
 resid=r[s].rolling(5,min_periods=5).sum()-beta*m.rolling(5,min_periods=5).sum()
 f[s]=-resid/(r[s].rolling(20,min_periods=15).std()*np.sqrt(252))
rows=[]
for t in f.index:
 for h in [1,5,10]:
  y=P.pct_change(h).shift(-h).loc[t]; z=pd.concat([f.loc[t],y],axis=1).dropna()
  if len(z)>=8: rows.append((t,h,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
df=pd.DataFrame(rows,columns=['date','h','ic','n']); print('range',P.index.min().date(),P.index.max().date(),'assets',len(U))
for h in [1,5,10]:
 q=df[df.h==h]; print(h,'dates',len(q),'meanIC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'avgN',q.n.mean())
print('coverage',f.notna().sum(axis=1).div(15).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01','2027-03-25')]:
 q=df[(df.h==1)&(df.date>=a)&(df.date<=b)]; print(name,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
f.stack().rename('signal').reset_index().rename(columns={'level_1':'symbol'}).to_csv('scripts/miner_2_20270325_residual_reversal_signal.csv',index=False)
