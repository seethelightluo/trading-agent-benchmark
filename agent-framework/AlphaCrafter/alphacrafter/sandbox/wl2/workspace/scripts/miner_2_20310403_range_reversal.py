import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=3000)
 if d is not None: D[s]=d.set_index('date')
cl=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill()
hi=pd.DataFrame({s:d.high.astype(float) for s,d in D.items()}).reindex(cl.index).ffill(); lo=pd.DataFrame({s:d.low.astype(float) for s,d in D.items()}).reindex(cl.index).ffill()
r=cl.pct_change(); rng=(hi-lo)/cl
# Range-normalized one-day reversal: penalize directional close moves by each
# instrument's contemporaneous true range, then neutralize common cross-asset move.
raw=-r/(rng.rolling(20,min_periods=10).median()+1e-8)
market=r.median(axis=1)
f=raw.sub(raw.median(axis=1),axis=0)
# suppress illiquid/noisy extremes with cross-sectional clipping
f=f.clip(lower=f.quantile(.05,axis=1),upper=f.quantile(.95,axis=1),axis=0)
rows=[]
for i in range(len(cl)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((cl.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for nm,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-31',a.index>='2026-01-01')]:
 z=a.loc[mask].ic; print(nm,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for h in [1,3,5,10]:
 rr=r.rolling(h).sum().shift(-h+1); vals=[]
 for i in range(len(cl)-h):
  z=pd.concat([f.iloc[i].rename('f'),rr.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append(z.f.corr(z.y))
 print('decay',h,round(np.mean(vals),6),len(vals))
f.to_csv('scripts/miner_2_20310403_range_reversal_signal.csv')
