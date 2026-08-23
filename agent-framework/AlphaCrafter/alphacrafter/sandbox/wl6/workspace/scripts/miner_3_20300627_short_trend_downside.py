import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2030-06-26'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]; R=P.pct_change(); neg=R.clip(upper=0).fillna(0)
down=np.sqrt((neg**2).rolling(20,min_periods=15).mean())*np.sqrt(252); cons=(R>0).rolling(10,min_periods=8).mean()
F=(P.pct_change(10)/(down+1e-12))*(0.5+cons); F=F.replace([np.inf,-np.inf],np.nan)
def calc(h):
 o=[]
 for i in range(25,len(P)-h):
  a=F.iloc[i].values;b=(P.iloc[i+h]/P.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:o.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(o,columns=['date','ic','n'])
x=calc(10);m=x.ic.mean();ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x));print(f'valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f} turnover_proxy {F.rank(axis=1,pct=True).diff().abs().mean().mean():.6f}')
for h in [5,20]:z=calc(h);print(f'decay_{h}d_ic {z.ic.mean():.8f} n {len(z)}')
print(x.assign(year=pd.to_datetime(x.date).dt.year).groupby('year').ic.agg(['mean','count']).to_string());print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())
