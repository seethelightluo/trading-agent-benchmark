import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is None:x=get_index_daily_data(s,4000)
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change(); v=r.rolling(20).std()
# Pullback continuation: buy recent dips within established medium trend, sell recent rallies in downtrend.
r5=p.pct_change(5); trend=p.pct_change(40); sig=(-r5/v*np.sign(trend)).replace([np.inf,-np.inf],np.nan).shift(1)
f=p.shift(-10)/p-1; rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
 if len(z)>=8:rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15);print('IC10',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean());print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20,40]:
 f=p.shift(-h)/p-1; rr=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(rr))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 z=q.loc[a:b,'ic'];print(a,b,len(z),z.mean(),z.mean()/z.std())
q.to_csv('scripts/miner_2_20340804_pullback_trend_signal.csv')
