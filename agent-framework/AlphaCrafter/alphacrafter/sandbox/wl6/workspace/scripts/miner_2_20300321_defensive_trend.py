import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-03-21'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); vol=R.rolling(30,min_periods=20).std()*np.sqrt(30)
neg=R.clip(upper=0).pow(2); down=np.sqrt(neg.rolling(30,min_periods=20).mean())*np.sqrt(30)
r30=P/P.shift(30)-1
F=(r30/(vol+1e-12))*(1/(1+down/(vol+1e-12)))
rows=[]
for i in range(70,len(P)-10):
 a=F.iloc[i].values; b=(P.iloc[i+10]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
 if ok.sum()>=8: rows.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']); ic=x.ic.mean(); ir=ic/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
rank=F.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).mean()
print(f'data_dates {len(P)} instruments {len(U)} horizon 10 valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.sum()/(len(x)*15):.5f} IC {ic:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f} turnover {turnover:.8f}')
print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
for h in [5,20]:
 z=[]
 for i in range(70,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:z.append(spearmanr(a[ok],b[ok]).statistic)
 print(f'decay_{h}d IC {np.mean(z):.8f} dates {len(z)}')
