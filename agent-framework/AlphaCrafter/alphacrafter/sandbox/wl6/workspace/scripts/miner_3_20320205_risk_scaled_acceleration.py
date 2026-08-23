import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2032-02-04')
cl={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}
C=pd.DataFrame(cl).loc[:cutoff]
r=C.pct_change(); vol=r.rolling(40,min_periods=20).std()
# acceleration: recent 20d return relative to 60d baseline, normalized by recent realized risk
F=(C/C.shift(20)-1 - (C/C.shift(60)-1)/3)/(vol*np.sqrt(20)+1e-12)

def calc(h):
 out=[]
 for i in range(61,len(C)-h):
  a=F.iloc[i]; b=C.iloc[i+h]/C.iloc[i]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   z=spearmanr(a[ok],b[ok]).statistic
   if np.isfinite(z): out.append((C.index[i],z,int(ok.sum())))
 return pd.DataFrame(out,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x)); print(f'h={h} dates={len(x)} avg_n={x.n.mean():.2f} coverage={x.n.mean()/15:.4f} IC={m:.7f} ICIR={ir:.4f} hit={(x.ic>0).mean():.4f}')
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
x=calc(10); print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string()); print('data',len(C),C.index.min().date(),C.index.max().date())
