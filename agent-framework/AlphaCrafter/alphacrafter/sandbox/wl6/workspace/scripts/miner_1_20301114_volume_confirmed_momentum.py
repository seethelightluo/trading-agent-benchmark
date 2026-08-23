import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-11-13'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.concat([D[s].close.rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
V=pd.concat([D[s].volume.rename(s) for s in U],axis=1).reindex(C.index)
R=C.pct_change(); volratio=V/(V.rolling(40,min_periods=20).median()+1e-12)
# momentum direction is trusted more when current activity exceeds its medium-term baseline
F=C.pct_change(10)*np.log1p(volratio.clip(lower=0))
def calc(h):
 out=[]
 for i in range(80,len(C)-h):
  a=F.iloc[i].values;b=(C.iloc[i+h]/C.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:out.append((C.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(out,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h);m=x.ic.mean();ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(252);print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
x=calc(10);print('turnover_proxy',F.rank(axis=1,pct=True).diff().abs().mean().mean());print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string());print('data_dates',len(C),'instruments',len(U),'data_end',C.index.max().date())
