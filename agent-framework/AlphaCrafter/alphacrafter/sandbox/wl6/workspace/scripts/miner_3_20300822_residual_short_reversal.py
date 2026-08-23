import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-08-21'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
# Residualized short reversal: remove each date's linear cross-sectional exposure to medium drawdown/rebound.
short=-(P/P.shift(3)-1)/(vol+1e-12)
medium=(P/P.rolling(60,min_periods=40).max()-1)/(vol+1e-12)
def resid(row,y):
 ok=np.isfinite(row)&np.isfinite(y)
 out=np.full(len(row),np.nan)
 if ok.sum()>=8:
  x=row[ok]; z=y[ok]
  A=np.column_stack([np.ones(len(x)),z]); beta=np.linalg.lstsq(A,x,rcond=None)[0]
  out[ok]=x-A@beta
 return out
F=pd.DataFrame([resid(short.iloc[i].values,medium.iloc[i].values) for i in range(len(P))],index=P.index,columns=U)
def calc(h):
 out=[]
 for i in range(70,len(P)-h):
  a=F.iloc[i].values;b=(P.iloc[i+h]/P.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: out.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(out,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x)); print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
x=calc(10); print(f'turnover_proxy {F.rank(axis=1,pct=True).diff().abs().mean().mean():.6f}')
print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())
