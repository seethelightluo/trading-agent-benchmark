import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2032-01-07')
cs={}; op={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index(); cs[s]=d.close; op[s]=d.open
C=pd.DataFrame(cs).loc[:cutoff]; O=pd.DataFrame(op).reindex(C.index)
gap=(O/C.shift(1)-1).rolling(5,min_periods=3).mean(); F=-gap.sub(gap.mean(axis=1),axis=0)
def calc(h):
 out=[]
 for i in range(10,len(C)-h):
  a=F.iloc[i]; b=C.iloc[i+h]/C.iloc[i]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   r=spearmanr(a[ok],b[ok]).statistic
   if np.isfinite(r): out.append((C.index[i],r,int(ok.sum())))
 return pd.DataFrame(out,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x)); print(h,len(x),x.n.mean(),x.n.mean()/15,m,ir,(x.ic>0).mean())
x=calc(10); print('turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean()); print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string()); print('data',len(C),C.index.min().date(),C.index.max().date())
