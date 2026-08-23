import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2032-01-21')
frames=[]
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()
 frames.append(d[['open','close','high','low']].rename(columns={c:f'{s}_{c}' for c in ['open','close','high','low']}))
D=pd.concat(frames,axis=1).loc[:cutoff]; C=pd.DataFrame({s:D[f'{s}_close'] for s in U}); O=pd.DataFrame({s:D[f'{s}_open'] for s in U}); H=pd.DataFrame({s:D[f'{s}_high'] for s in U}); L=pd.DataFrame({s:D[f'{s}_low'] for s in U})
rng=(H-L).replace(0,np.nan)
pressure=((C-O)/rng).rolling(10,min_periods=8).mean()
F=-(pressure.sub(pressure.mean(axis=1),axis=0))
def calc(h):
 z=[]
 for i in range(15,len(C)-h):
  a=F.iloc[i].values; b=(C.iloc[i+h]/C.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: z.append((C.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(z,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x)); print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
print('turnover_proxy',F.rank(axis=1,pct=True).diff().abs().mean().mean()); x=calc(10); print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string()); print('data_dates',len(C),'instruments',len(U),'data_end',C.index.max().date())
