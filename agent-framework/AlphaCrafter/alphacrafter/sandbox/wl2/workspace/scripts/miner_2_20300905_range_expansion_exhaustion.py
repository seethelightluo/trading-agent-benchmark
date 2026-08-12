import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=3000)
 if x is not None and len(x):D[s]=x.set_index('date')
q={k:v['close'].astype(float) for k,v in D.items()}; p=pd.DataFrame(q).sort_index().ffill(); r=p.pct_change()
# Range-expansion exhaustion: reverse relative 3-day move, amplified only when today's true range expands versus its 20d baseline.
h={k:v['high'].astype(float) for k,v in D.items()}; l={k:v['low'].astype(float) for k,v in D.items()}; hi=pd.DataFrame(h).reindex(p.index).ffill(); lo=pd.DataFrame(l).reindex(p.index).ffill()
tr=(hi-lo).div(p).replace([np.inf,-np.inf],np.nan); expansion=(tr/(tr.rolling(20,min_periods=10).median()+1e-12)).clip(0,4)
med=r.median(axis=1); resid=r.sub(med,axis=0); gate=(expansion-1).clip(lower=0)
f=-resid.rolling(3,min_periods=3).sum()* (1+gate)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); x=a.ic
print('dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),z.mean(),z.mean()/z.std(ddof=1), (z>0).mean())
for hzn in [1,3,5]:
 rr=p.pct_change(hzn).shift(-hzn)
 vals=[]
 for i in range(len(p)-hzn):
  z=pd.concat([f.iloc[i].rename('f'),rr.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append(z.f.corr(z.y))
 print('horizon',hzn,'IC',np.nanmean(vals),'n',len(vals))
f.to_csv('scripts/miner_2_20300905_range_expansion_exhaustion_signal.csv')
