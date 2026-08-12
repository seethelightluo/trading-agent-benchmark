import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in assets:
 d=get_stock_daily_data(s,2200)
 if d is None or len(d)<150:d=get_index_daily_data(s,2200)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);fs[s]=d.drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(fs).sort_index(); r=p.pct_change();
# Relative strength residualized against contemporaneous cross-asset median at each lookback
m20=r.rolling(20).sum(); m60=r.rolling(60).sum(); med20=m20.median(axis=1); med60=m60.median(axis=1)
# test quality of relative medium trend, lagged by one completed day
sig=(m20-med20.values[:,None])-(m60-med60.values[:,None])/3
sig=sig.shift(1)
rows=[]
for dt,x in sig.iterrows():
 n=x.notna()
 if n.sum()<8:continue
 rec={'date':dt,'n':n.sum()}
 for h in [1,5,10,20]:
  y=(p.shift(-h)/p-1).loc[dt]; z=y[n].dropna();xx=x[z.index]
  if len(z)>=8:rec['ic'+str(h)]=xx.corr(z)
 rows.append(rec)
o=pd.DataFrame(rows).set_index('date');print('assets',len(fs),'dates',len(o),'avg_n',o.n.mean())
for h in [1,5,10,20]:
 q=o['ic'+str(h)].dropna();print(h,'IC %.6f ICIR %.6f hit %.4f obs %d'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
print('annual')
for y,g in o.groupby(o.index.year):
 q=g.ic1.dropna();print(y,len(q),'%.5f %.5f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
sig.to_csv('scripts/miner_3_20300905_relative_strength_signal.csv')
