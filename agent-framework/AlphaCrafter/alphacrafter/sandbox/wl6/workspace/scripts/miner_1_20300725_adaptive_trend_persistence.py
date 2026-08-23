import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2030-07-24')
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); vol=R.rolling(40,min_periods=25).std()*np.sqrt(252)
r10=P.pct_change(10); r30=P.pct_change(30); r90=P.pct_change(90)
pos30=(R>0).rolling(30,min_periods=20).mean()
# Adaptive persistence trend: medium/long trend, breadth persistence, and total-risk scaling.
F=(0.6*r30+0.4*r90)*(0.5+0.5*pos30)/(vol+1e-12)
F=F.replace([np.inf,-np.inf],np.nan)
def calc(h):
 out=[]
 for i in range(95,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: out.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(out,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
 print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
x=calc(10); print(f'turnover_proxy {F.rank(axis=1,pct=True).diff().abs().mean().mean():.6f}')
print(x.assign(year=pd.to_datetime(x.date).dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())
