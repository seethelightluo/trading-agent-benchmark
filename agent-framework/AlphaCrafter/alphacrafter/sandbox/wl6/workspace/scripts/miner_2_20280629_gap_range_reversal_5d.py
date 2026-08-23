import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data')
raw={a:pd.read_csv(base/(a+'.csv'),parse_dates=['date']).set_index('date') for a in A}
# Gap/range-aware reversal: recent close-to-close loss, normalized by trailing true-range volatility.
p=pd.DataFrame({a:d['close'] for a,d in raw.items()}).sort_index().ffill().loc[:'2028-06-28']
h=pd.DataFrame({a:d['high'] for a,d in raw.items()}).reindex(p.index).ffill(); l=pd.DataFrame({a:d['low'] for a,d in raw.items()}).reindex(p.index).ffill()
prev=p.shift(1); tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).groupby(level=0,axis=1).max() if False else pd.DataFrame({a:pd.concat([h[a]-l[a],(h[a]-prev[a]).abs(),(l[a]-prev[a]).abs()],axis=1).max(axis=1) for a in A})
vol=(tr/p).rolling(20).mean()
f=(-(p.pct_change().rolling(5).sum())/(vol+1e-8)).shift(1)
print('sample',p.index.min().date(),p.index.max().date(),'assets',p.shape[1])
for horizon in [1,5,10]:
 y=p.shift(-horizon).div(p)-1; vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 s=pd.Series(vals,index=dates).dropna(); print('horizon',horizon,'dates',len(s),'avg_n',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),6),'hit',round((s>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for nm,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-25':('2023','2025-12-31'),'2026-27':('2026','2027-12-31'),'2028':('2028','2028-06-28')}.items():
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],(p.shift(-1).div(p)-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(vals).dropna(); print('regime',nm,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),6),'hit',round((s>0).mean(),4))
