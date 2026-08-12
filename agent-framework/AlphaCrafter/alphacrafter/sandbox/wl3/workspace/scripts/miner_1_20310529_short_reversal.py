import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except:pass
  if x is not None and len(x):break
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();
# short-horizon residual reversal, volatility damped, with 1d shock cap
r1=r; rv=r.rolling(20).std(); res=r1.sub(r1.median(axis=1),axis=0)
f=(-res/(rv+1e-12)).clip(-5,5)
for h in [3,5,6,10]:
 a=[]
 for i in range(len(p)-h-1):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   c=z.f.corr(z.y)
   if np.isfinite(c):a.append((p.index[i],c,len(z)))
 a=pd.DataFrame(a,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
  q=a.loc[lo:hi];print(lo, len(q),q.ic.mean() if len(q) else np.nan)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()/2)
f.index.name='date';f.reset_index().to_csv('scripts/miner_1_20310529_short_reversal_signal.csv',index=False)
