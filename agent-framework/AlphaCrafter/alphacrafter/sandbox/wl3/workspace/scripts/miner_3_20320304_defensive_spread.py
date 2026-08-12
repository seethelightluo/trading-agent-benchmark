import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except Exception:x=None
  if x is not None and len(x):break
 if x is not None and len(x):D[s]=x.assign(date=pd.to_datetime(x.date)).sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=np.log(pd.DataFrame(D).sort_index().ffill());r=p.diff();r20=r.rolling(20).sum(); risk=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','COPPER','WTI','BTC','ETH']; defs=['XAU','US10Y','CN10Y']; f=(r20.sub(r20[risk].mean(axis=1),axis=0)-r20.sub(r20[defs].mean(axis=1),axis=0)*.25).shift(1); fr=p.shift(-10)-p
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8:rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [60,120,252]:
 x=q.tail(n);print('recent',n,x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_3_20320304_defensive_spread_signal.csv');z.to_csv('scripts/miner_3_20320304_defensive_spread_ic.csv')
