import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-05-16'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
# Recent momentum relative to its longer-run average: acceleration, risk scaled.
F=(P.pct_change(20)-P.pct_change(60)/3)/(vol+1e-12)
rows=[]
for i in range(70,len(P)-10):
 a=F.iloc[i].values; b=(P.iloc[i+10]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
 if ok.sum()>=8: rows.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
print(f'valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.sum()/(len(x)*15):.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f} turnover_proxy {F.rank(axis=1,pct=True).diff().abs().mean().mean():.6f}')
for h in [5,20]:
 z=[]
 for i in range(70,len(P)-h):
  a=F.iloc[i].values;b=(P.iloc[i+h]/P.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:z.append(spearmanr(a[ok],b[ok]).statistic)
 print(f'decay_{h}d_ic {np.mean(z):.8f} n {len(z)}')
print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())
