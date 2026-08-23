import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-08-07'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
R=P.pct_change(); vol=R.rolling(30,min_periods=20).std()*np.sqrt(20)
# Trend signal strengthened in calm regimes and damped in volatility shocks; macro multiplier is common, not future-looking.
r20=P/P.shift(20)-1
vix_z=(vix-vix.rolling(120,min_periods=60).mean())/(vix.rolling(120,min_periods=60).std()+1e-12)
macro=(1-0.30*np.clip(vix_z,-2,2))
F=r20/(vol+1e-12)*macro.values[:,None]
def calc(h):
 rows=[]
 for i in range(130,len(P)-h):
  a=F.iloc[i].values;b=(P.iloc[i+h]/P.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: rows.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(rows,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h);m=x.ic.mean();ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x));print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
x=calc(10);print('turnover_proxy',F.rank(axis=1,pct=True).diff().abs().mean().mean());print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string());print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())
