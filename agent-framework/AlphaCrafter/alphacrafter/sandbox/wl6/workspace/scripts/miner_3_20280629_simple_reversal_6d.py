import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def f(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100:d=get_index_daily_data(s,4000)
 return d
c=pd.DataFrame({s:f(s).set_index('date').close for s in U}).sort_index(); r=c.pct_change(); sig=-r.rolling(6,min_periods=6).sum().shift(1)
for h in [1,5,10]:
 z=[]
 for dt in sig.index:
  x=pd.concat([sig.loc[dt],c.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(x)>=8:z.append((dt,x.iloc[:,0].corr(x.iloc[:,1]),len(x)))
 z=pd.DataFrame(z,columns=['date','ic','n']).set_index('date'); m=z.ic.mean();sd=z.ic.std(ddof=1)
 print(f'H={h} dates={len(z)} avgN={z.n.mean():.2f} IC={m:.8f} ICIR={m/sd*np.sqrt(len(z)):.8f} hit={(z.ic>0).mean():.4f}')
 if h==1:
  for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2028')]:
   q=z.loc[a:b]
   if len(q):print(f'regime {a}-{b} dates={len(q)} IC={q.ic.mean():.8f}')
print('coverage',sig.notna().sum(axis=1).mean()/15,'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'instruments',c.shape[1])
