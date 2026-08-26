import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl, hi, lo = {}, {}, {}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  q=d[['date','close','high','low']].copy(); q.date=pd.to_datetime(q.date); q=q.drop_duplicates('date').set_index('date').sort_index()
  cl[s]=q.close.astype(float); hi[s]=q.high.astype(float); lo[s]=q.low.astype(float)
p=pd.DataFrame(cl).sort_index(); H=pd.DataFrame(hi).reindex(p.index); L=pd.DataFrame(lo).reindex(p.index)
r=p.pct_change(); r20=p.pct_change(20); vol=r.rolling(30).std()
# Recent true range relative to its 90-day baseline; stressed markets receive stronger contrarian weight.
range20=((H-L)/p).rolling(20).mean(); range90=range20.rolling(90).median()
stress=(range20/range90).clip(0.5,2.0)
f=(-r20/vol * stress).shift(1)
allz={}
for h in [5,10,20,40,60]:
 fr=p.shift(-h).div(p)-1; vals=[]; dates=[]; counts=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8 and a.iloc[:,0].nunique()>1 and a.iloc[:,1].nunique()>1:
   vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(dt); counts.append(len(a))
 z=pd.Series(vals,index=pd.to_datetime(dates)).dropna(); allz[h]=z
 print(f'h={h} dates={len(z)} avgN={np.mean(counts):.2f} coverage={np.mean(counts)/len(U):.4f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1)*np.sqrt(len(z)):.6f} hit={np.mean(z>0):.4f}')
z=allz[20]
for lo_,hi_,nm in [('2024-01-01','2026-12-31','2024-26'),('2027-01-01','2029-12-31','2027-29'),('2030-01-01','2030-11-27','2030YTD')]:
 q=z.loc[lo_:hi_]; print(f'regime={nm} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 else np.nan:.6f}')
print(f'loaded={len(cl)} dates={len(p)} avgN={f.notna().sum(axis=1).mean():.2f} turnover={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f}')
f.to_csv('scripts/miner_3_20301128_range_stressed_reversal_signal.csv',index_label='date')
