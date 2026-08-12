import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in A:
 d=get_stock_daily_data(s,2200)
 if d is None or len(d)<150: d=get_index_daily_data(s,2200)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);fs[s]=d.drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(fs).sort_index(); r=p.pct_change()
# Trend quality: medium return weighted by directional efficiency and persistence; all inputs lagged one day.
r30=p.pct_change(30); abs_sum=r.abs().rolling(30).sum(); eff=r30.abs()/abs_sum
up=r.gt(0).rolling(30).mean(); down=r.lt(0).rolling(30).mean()
# signed trend, reward coherent moves and penalize two-sided paths
sig=(r30/(r.abs().rolling(30).mean()+1e-8))*eff*(0.5+0.5*(up-down).abs())
sig=sig.shift(1)
rows=[]
for dt,x in sig.iterrows():
 n=x.notna()
 if n.sum()<8: continue
 rec={'date':dt,'n':int(n.sum())}
 for h in [1,5,10,20]:
  y=(p.shift(-h)/p-1).loc[dt]; z=y[n].dropna(); xx=x[z.index]
  if len(z)>=8 and xx.nunique()>1 and z.nunique()>1: rec['ic'+str(h)]=xx.corr(z)
 rows.append(rec)
o=pd.DataFrame(rows).set_index('date');print('assets',len(fs),'dates',len(o),'avg_n',o.n.mean())
for h in [1,5,10,20]:
 q=o['ic'+str(h)].dropna();print(h,'IC %.6f ICIR %.6f hit %.4f obs %d'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
print('annual')
for y,g in o.groupby(o.index.year):
 q=g.ic1.dropna();print(y,len(q),'%.5f %.5f'%(q.mean(),q.mean()/q.std(ddof=1) if q.std()>0 else np.nan))
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
sig.to_csv('scripts/miner_3_20300905_trend_quality_signal.csv')
# signal provenance metadata
print('artifact','scripts/miner_3_20300905_trend_quality_signal.csv')
