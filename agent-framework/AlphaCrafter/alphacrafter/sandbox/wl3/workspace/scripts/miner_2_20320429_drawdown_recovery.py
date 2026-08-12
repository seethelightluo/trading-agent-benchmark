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
p=np.log(pd.DataFrame(D).sort_index().ffill()); r=p.diff(); peak=p.rolling(60,min_periods=20).max(); dd=(p-peak).clip(upper=0)
# Drawdown-recovery: rebound from the 60d trough, scaled by current drawdown severity.
rebound=p-p.rolling(20,min_periods=10).min(); raw=rebound/(dd.abs()+0.02)
f=raw.rolling(5,min_periods=5).mean().shift(1); fr=p.shift(-10)-p; rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8:rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('dates',len(z),'avgN',round(z.n.mean(),2),'assets',len(D),'coverage',round(z.n.mean()/len(D),4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
for n in [120,252,756]:
 x=q.tail(n);print('recent',n,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b];print(a,b,'n',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
f.to_csv('scripts/miner_2_20320429_drawdown_recovery_signal.csv');z.to_csv('scripts/miner_2_20320429_drawdown_recovery_ic.csv')
