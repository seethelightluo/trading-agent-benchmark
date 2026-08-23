import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-06-27'); base='../persistent/stock_data'
ps={}; vs={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()
 ps[s]=d['close']; vs[s]=d['volume'] if 'volume' in d else pd.Series(index=d.index,dtype=float)
P=pd.DataFrame(ps).sort_index().loc[:cutoff]; V=pd.DataFrame(vs).reindex(P.index)
R=P.pct_change(); rv=R.rolling(20,min_periods=15).std()*np.sqrt(252)
vol_ratio=V.rolling(5,min_periods=3).mean()/(V.rolling(20,min_periods=10).mean()+1e-12)
F=(P.pct_change(10)/(rv+1e-12))*vol_ratio
F=F.replace([np.inf,-np.inf],np.nan)
def calc(h):
 out=[]
 for i in range(25,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: out.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(out,columns=['date','ic','n'])
x=calc(10); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
print(f'valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f} turnover_proxy {F.rank(axis=1,pct=True).diff().abs().mean().mean():.6f}')
for h in [5,20]:
 z=calc(h); print(f'decay_{h}d_ic {z.ic.mean():.8f} n {len(z)}')
print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())
