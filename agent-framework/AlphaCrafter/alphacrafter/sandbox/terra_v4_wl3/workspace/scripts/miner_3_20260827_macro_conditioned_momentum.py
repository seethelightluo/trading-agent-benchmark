import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data')
def load(p):
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); return d.set_index('date')['close'].astype(float)
px=pd.concat({s:load(base/(s+'.csv')) for s in U},axis=1).sort_index()
v=load(Path('../persistent/index_data/VIX.csv')).reindex(px.index).ffill()
regime=-(v.pct_change(5)).clip(-.5,.5).fillna(0.0)
mom=px.pct_change(20); f=mom.mul(1.0+regime,axis=0)
f=f.loc[:'2026-07-15']; px=px.loc[f.index]
def calc(h):
 y=px.shift(-h)/px-1; vals=[]; dates=[]; names=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   r=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(r): vals.append(r); dates.append(dt); names.append(len(z))
 return pd.Series(vals,index=pd.DatetimeIndex(dates)), names
print('candidate macro_conditioned_momentum_20d')
print('range',f.index.min(),f.index.max(),'assets',len(U),'dates',len(f.index))
for h in [1,5,10,20]:
 a,n=calc(h); print(h,'dates',len(a),'avg_names',round(np.mean(n),2) if n else 0,'meanIC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
a,n=calc(1)
print('annual',a.groupby(a.index.year).agg(['mean','count']).round(4).to_dict())
print('coverage',round(f.notna().sum().sum()/f.size,4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
print('regime_split',a.groupby(regime.reindex(a.index).gt(0).map({True:'vix_down',False:'vix_up'})).agg(['mean','count']).round(4).to_dict())
