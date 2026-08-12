import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except Exception:x=None
  if x is not None and len(x):break
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); lp=np.log(p); r=lp.diff()
mom=lp-lp.shift(60); vol=np.sqrt((r**2).rolling(20).mean())*np.sqrt(252)
peak=p.rolling(120).max(); trough=p.rolling(120).min(); rec=(p-trough)/(peak-trough).replace(0,np.nan)
f=(mom/vol*rec).replace([np.inf,-np.inf],np.nan).shift(1)
fr=lp.shift(-10)-lp; rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('shape',p.shape,'valid_dates',len(z),'assets',len(D),'avgN',z.n.mean(),'coverage',z.n.mean()/len(D))
print('H10 IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2032')]:
 x=q.loc[lo:hi];print(lo,len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>2 else np.nan)
for n in [60,120,252]:
 x=q.tail(n);print('recent',n,len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>2 else np.nan)
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_3_20320219_recovery_momentum60_signal.csv');z.to_csv('scripts/miner_3_20320219_recovery_momentum60_ic.csv')
