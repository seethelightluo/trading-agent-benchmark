import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is None:x=get_index_daily_data(s,4000)
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();down=r.where(r<0,0).rolling(20).std()
# Short/medium trend quality: recent 20d return per unit downside risk,
# lagged to keep the signal causal.
sig=(p.pct_change(20)/(down+1e-8)).shift(1)
rows=[]
for d in sig.index:
 f=p.shift(-10).loc[d]/p.loc[d]-1;z=pd.concat([sig.loc[d],f],axis=1).dropna()
 if len(z)>=8:rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15)
print('IC10',q.ic.mean(),'ICIRdaily',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20,40]:
 a=[]
 for d in sig.index:
  f=p.shift(-h).loc[d]/p.loc[d]-1;z=pd.concat([sig.loc[d],f],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(a))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 z=q.loc[a:b,'ic'];print('regime',a,b,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std())
sig.to_csv('scripts/miner_3_20341208_short_downside_momentum_signal.csv',index_label='date');q.to_csv('scripts/miner_3_20341208_short_downside_momentum_ic.csv')
