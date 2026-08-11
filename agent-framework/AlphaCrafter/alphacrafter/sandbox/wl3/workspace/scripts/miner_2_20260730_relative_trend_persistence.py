import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
R=pd.concat({s:D[s].close.pct_change() for s in U},axis=1).sort_index(); med=R.median(axis=1); resid=R.sub(med,axis=0)
F=resid.rolling(10,min_periods=7).sum()/resid.rolling(20,min_periods=10).std()
rows=[]
for s in U: rows.append(pd.DataFrame({'date':F.index,'f':F[s],'y':R[s].shift(-1)}))
A=pd.concat(rows,ignore_index=True).dropna(); obs=[]
for dt,g in A.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: obs.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
O=pd.DataFrame(obs,columns=['date','ic','n']).set_index('date'); x=O.ic
turnover=F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(); coverage=len(A)/(len(R.index)*15)
print('cutoff',cut.date(),'dates',len(O),'avg_n',O.n.mean(),'coverage',coverage,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turnover',turnover)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 z=O.loc[lo:hi].ic;print('regime',lo,hi,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
for h in [5,10]:
 B=[]
 for s in U: B.append(pd.DataFrame({'date':F.index,'f':F[s],'y':R[s].rolling(h).sum().shift(-h)}))
 B=pd.concat(B,ignore_index=True).dropna(); z=[]
 for dt,g in B.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:z.append(spearmanr(g.f,g.y).statistic)
 print('decay',h,'dates',len(z),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1))
