import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-07-10'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); vol=R.rolling(20,min_periods=15).std()*np.sqrt(252)
# Agreement-weighted medium momentum: short/medium/long returns, with bonus only when signs agree.
r5=P.pct_change(5); r20=P.pct_change(20); r60=P.pct_change(60)
base_mom=(0.25*r5+0.50*r20+0.25*r60)/(vol+1e-12)
agree=((np.sign(r5)==np.sign(r20)) & (np.sign(r20)==np.sign(r60))).astype(float)
F=base_mom*(0.5+0.5*agree)
F=F.replace([np.inf,-np.inf],np.nan)
def calc(h):
 out=[]
 for i in range(65,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: out.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(out,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
 print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
x=calc(10); print(f'turnover_proxy {F.rank(axis=1,pct=True).diff().abs().mean().mean():.6f}')
print(x.assign(year=pd.to_datetime(x.date).dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())
