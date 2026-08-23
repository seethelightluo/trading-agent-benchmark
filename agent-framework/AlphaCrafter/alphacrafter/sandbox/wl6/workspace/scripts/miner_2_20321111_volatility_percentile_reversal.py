import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-11-10']
r=p.pct_change()
# Higher recent volatility percentile receives stronger reversal weight: recent 20d return
# divided by its 252d cross-sectional time-series volatility percentile, with low-vol assets retained.
rv=r.rolling(20).std()
rank=rv.rolling(252,min_periods=126).rank(pct=True)
sig=-r.rolling(20).sum()*(0.5+rank)
print('candidate=volatility-percentile-weighted 20d reversal; universe',len(syms),'data_dates',len(p))
for h in [5,10,20,40]:
 f=p.shift(-h).div(p)-1; vals=[]; ns=[]; ds=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 a=np.asarray(vals); print('horizon',h,'dates',len(a),'avg_n',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 if h==20:
  print('coverage',np.mean([len(z) for z in []]) if False else sig.notna().sum().sum()/sig.size)
  print('year_IC',pd.Series(a,index=ds).groupby(pd.Index(ds).year).mean().round(5).to_dict())
