import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-09-29']; r=p.pct_change()
# Observation-only VIX regime: reversal is stronger when VIX is above its trailing median.
v=pd.read_csv(Path('../persistent/index_data/VIX.csv')); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.astype(float).reindex(p.index).ffill()
stress=(v>v.rolling(120,min_periods=60).median()).astype(float)
base=-r.rolling(20).sum(); sig=base.mul(1+0.5*stress,axis=0)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],p.shift(-10).div(p).sub(1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=a.ic
print('candidate=VIX-conditioned 20d reversal; horizon=10d');print('dates',len(a),'avg_n',a.n.mean(),'coverage',sig.notna().sum().sum()/sig.size)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for h in [5,10,20,40]:
 fw=p.shift(-h).div(p)-1; q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('decay',h,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
a['year']=a.index.year;print('year_IC');print(a.groupby('year').ic.mean().to_string())
