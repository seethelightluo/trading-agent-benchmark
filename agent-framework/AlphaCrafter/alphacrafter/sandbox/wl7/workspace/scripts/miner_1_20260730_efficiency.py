import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
P={}; F={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index(); d=d[d.index<=end]; P[s]=d
 ret=d.pct_change(); F[s]=(d/d.shift(20)-1)/ret.abs().rolling(20,min_periods=15).sum()
f=pd.DataFrame(F); p=pd.DataFrame(P)
for h in [1,5,10]:
 vals=[]
 for dt in f.index:
  a=[]
  for s in U:
   if dt in F[s].index:
    j=F[s].index.get_loc(dt)
    if j+h<len(P[s]): a.append((F[s].iloc[j],P[s].iloc[j+h]/P[s].iloc[j]-1))
  z=pd.DataFrame(a).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z[0],z[1]).statistic,len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print('h',h,'dates',len(q),'mean_n',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'coverage',q.n.mean()/15)
 for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
  x=q.loc[a:b].ic; print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
# cross-sectional rank turnover on common date values
ranks=f.rank(axis=1,pct=True); print('rank turnover',(ranks-ranks.shift()).abs().mean(axis=1).mean())
print('factor valid',f.notna().sum(axis=1).mean())
