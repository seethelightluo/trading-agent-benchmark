import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: x=fn(s,days=5000)
  except Exception: x=None
  if x is not None and len(x): break
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
ret5=np.log(p/p.shift(5)); vol20=r.rolling(20).std()*np.sqrt(252)
breadth=(ret5>0).mean(axis=1); stress=(breadth<0.40)
f=(-ret5/vol20).where(stress).shift(1)
fr=np.log(p.shift(-10)/p)
rows=[]
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum(),int(stress.loc[dt])))
z=pd.DataFrame(rows,columns=['date','ic','n','stress']).set_index('date'); ics=z.ic
print('shape',p.shape,'dates',len(z),'assets',len(D),'avgN',z.n.mean(),'coverage',z.n.mean()/len(D))
print('H10 IC %.8f ICIR %.8f hit %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1),(ics>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=ics.loc[lo:hi]; print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>2 else np.nan)
q=ics.tail(120); print('recent120',q.mean(),q.mean()/q.std(ddof=1) if len(q)>2 else np.nan)
print('stress_count',int(stress.sum()),'stress_ic',z.loc[z.stress==1,'ic'].mean())
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_3_20310821_stress_reversal_signal.csv')
