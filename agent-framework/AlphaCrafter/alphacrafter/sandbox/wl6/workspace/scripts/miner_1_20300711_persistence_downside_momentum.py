import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-07-11'); base='../persistent/stock_data'
ps={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index(); ps[s]=d['close']
P=pd.DataFrame(ps).sort_index().loc[:cutoff]; R=P.pct_change()
down=R.where(R<0,0).rolling(30,min_periods=20).std()*np.sqrt(252)
persistence=(R>0).rolling(20,min_periods=15).mean()
# medium momentum rewarded only when direction is persistent, risk measured by downside variation
F=(P.pct_change(10)/(down+1e-12))*persistence
F=F.replace([np.inf,-np.inf],np.nan)
def calc(h):
 out=[]
 for i in range(35,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: out.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(out,columns=['date','ic','n'])
x=calc(10); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
print(f'valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f} turnover_proxy {F.rank(axis=1,pct=True).diff().abs().mean().mean():.6f}')
for h in [5,20]:
 z=calc(h); print(f'decay_{h}d_ic {z.ic.mean():.8f} n {len(z)}')
print('regimes'); print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())
