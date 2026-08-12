import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   z=fn(s,days=5000)
   if z is not None and len(z): x=z; break
  except: pass
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').sort_index()
S={}
for s,x in D.items():
 c=pd.to_numeric(x.close,errors='coerce'); r=np.log(c/c.shift(1)); net=np.log(c/c.shift(20)); path=r.abs().rolling(20,min_periods=15).sum()
 # efficient trend, mildly volatility scaled
 S[s]=pd.DataFrame({'f':(net/path).clip(-1,1),'ret':np.log(c.shift(-10)/c)})
rows=[]
for d in sorted(set().union(*[set(x.index) for x in S.values()])):
 a=[(s,z.loc[d,'f'],z.loc[d,'ret']) for s,z in S.items() if d in z.index and np.isfinite(z.loc[d,'f']) and np.isfinite(z.loc[d,'ret'])]
 if len(a)>=8:
  q=pd.DataFrame(a,columns=['s','f','r']); rows.append((d,q.f.corr(q.r,method='spearman'),len(a)))
q=pd.DataFrame(rows,columns=['date','ic','n']); v=q.ic.dropna(); print('dates',len(v),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2031')]:
 z=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b+'-12-31')].ic;print(a,b,len(z),z.mean(),z.mean()/z.std(ddof=1))
out=[]
for d in q.date:
 for s,z in S.items():
  if d in z.index and np.isfinite(z.loc[d,'f']):out.append({'date':d,'symbol':s,'signal':z.loc[d,'f']})
pd.DataFrame(out).to_csv('scripts/miner_1_20311127_efftrend_signal.csv',index=False)
