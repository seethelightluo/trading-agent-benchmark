import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-04-18'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); lo=P.rolling(60,min_periods=30).min(); hi=P.rolling(60,min_periods=30).max(); pos=(P-lo)/(hi-lo+1e-12)
ret20=P/P.shift(20)-1; vol20=R.rolling(20,min_periods=15).std()
F=-((pos-.5)*2 + ret20/(vol20*np.sqrt(20)+1e-12))/2
for h in [5,10,20]:
 rows=[]
 for i in range(80,len(P)-h):
  a=F.iloc[i].values;b=(P.iloc[i+h]/P.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: rows.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 x=pd.DataFrame(rows,columns=['date','ic','n']); ic=x.ic.mean(); ir=ic/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
 print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.sum()/(len(x)*15):.5f} IC {ic:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
 if h==10: print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('data_dates',len(P),'instruments',len(U))
