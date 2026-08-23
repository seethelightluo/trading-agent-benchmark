import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-05-30'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); F=-(P.pct_change(5))/(R.rolling(20,min_periods=15).std()*np.sqrt(252)+1e-12)
def run(h):
 rows=[]
 for i in range(25,len(P)-h):
  a=F.iloc[i].values;b=(P.iloc[i+h]/P.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: rows.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(rows,columns=['date','ic','n'])
x=run(10); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
print(f'valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.sum()/(len(x)*15):.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f} turnover_proxy {F.rank(axis=1,pct=True).diff().abs().mean().mean():.6f}')
for h in [5,20]:
 z=run(h); print(f'decay_{h}d_ic {z.ic.mean():.8f} n {len(z)}')
print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())
