import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-07-15'); D={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); D[s]=d[d.index<=end]
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index(); macro=macro[macro.index<=end].close.pct_change()
# defensive DXY beta: negative rolling covariance with DXY returns
F={}
for s,d in D.items():
 r=d.close.pct_change(); x=pd.concat([r,macro],axis=1).dropna(); cov=x.iloc[:,0].rolling(60,min_periods=45).cov(x.iloc[:,1]); var=macro.rolling(60,min_periods=45).var(); F[s]=-(cov/var).reindex(d.index)
for h in [1,5,10]:
 out=[]
 for dt in sorted(set.intersection(*[set(d.index) for d in D.values()])):
  a=[]
  for s,d in D.items():
   if dt in F[s].index:
    j=d.index.get_loc(dt)
    if j+h<len(d): a.append((F[s].loc[dt],d.close.iloc[j+h]/d.close.iloc[j]-1))
  z=pd.DataFrame(a).dropna()
  if len(z)>=8: out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
 print('h',h,'dates',len(q),'mean_n',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'coverage',q.n.mean()/15)
 for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
  x=q.loc[a:b].ic; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
ranks=pd.DataFrame(F).rank(pct=True); print('turnover',(ranks-ranks.shift()).abs().mean(axis=1).mean())
print('valid',pd.DataFrame(F).notna().sum(axis=1).mean())
# recent half
x=q.loc['2025':'2026'].ic; print('recent',len(x),x.mean(),x.mean()/x.std(ddof=1))
