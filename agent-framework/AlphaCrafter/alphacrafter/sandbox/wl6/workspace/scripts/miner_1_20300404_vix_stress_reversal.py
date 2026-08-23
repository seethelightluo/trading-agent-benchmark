import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-04-04'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
P=P.loc[~P.index.duplicated(keep='last')]
R=P.pct_change(); vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']; vix=vix[~vix.index.duplicated(keep='last')].reindex(P.index).ffill()
vp=vix.rolling(120,min_periods=60).rank(pct=True)
# Stress-conditioned reversal: short-term losers receive more reversal weight during elevated VIX.
F=(-R.rolling(5,min_periods=5).sum().div(vol)) .mul(1+0.75*vp,axis=0)
for h in [5,10,20]:
 rows=[]
 for i in range(130,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: rows.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 x=pd.DataFrame(rows,columns=['date','ic','n']); ic=x.ic.mean(); ir=ic/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
 print(f'horizon {h} data_dates {len(P)} instruments {len(U)} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.sum()/(len(x)*15):.5f} IC {ic:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
 print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
