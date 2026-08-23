import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2030-08-21'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); v=R.rolling(20,min_periods=15).std()*np.sqrt(20)
# rebound from 60-day low, volatility normalized
F=((P/P.rolling(60,min_periods=40).min()-1)/(v+1e-12)).replace([np.inf,-np.inf],np.nan)
def calc(h):
 o=[]
 for i in range(65,len(P)-h):
  a=F.iloc[i].values;b=(P.iloc[i+h]/P.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:o.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(o,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h);m=x.ic.mean();print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x)):.5f} hit {(x.ic>0).mean():.5f}')
x=calc(10);print(f'turnover_proxy {F.rank(axis=1,pct=True).diff().abs().mean().mean():.6f}');print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string());print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())