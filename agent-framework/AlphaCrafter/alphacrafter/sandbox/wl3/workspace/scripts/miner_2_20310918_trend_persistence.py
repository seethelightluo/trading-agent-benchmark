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
p=pd.DataFrame(D).sort_index().ffill();r=np.log(p).diff()
# Trend persistence: volatility-scaled 20d return multiplied by fraction of positive sessions, all lagged.
rv=r.rolling(30).std()*np.sqrt(252)
positive=(r>0).rolling(20).mean()
f=(np.log(p/p.shift(20))/rv*positive).shift(1)
for h in (1,5,10,20):
 y=np.log(p.shift(-h)/p); rows=[]
 for dt in f.index:
  a,b=f.loc[dt],y.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1:rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
 print('H',h,'dates',len(z),'avgN',round(z.n.mean(),3),'IC',round(q.mean(),8),'ICIR',round(q.mean()/q.std(ddof=1),8),'hit',round((q>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
  a=q.loc[lo:hi];print(lo,len(a),round(a.mean(),7),round(a.mean()/a.std(ddof=1),5))
 a=q.tail(120);print('recent120',round(a.mean(),7),round(a.mean()/a.std(ddof=1),5))
print('coverage',round(f.notna().sum(axis=1).mean()/len(D),5),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
f.to_csv('scripts/miner_2_20310918_trend_persistence_signal.csv')
