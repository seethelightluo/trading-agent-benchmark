import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof=pd.Timestamp('2030-02-07'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in syms:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<150:d=get_index_daily_data(s,2800)
 if d is not None and len(d):
  d=d[d.date<=asof].set_index('date').sort_index();p[s]=d.close.astype(float)
c=pd.DataFrame(p); r=c.pct_change(); t=c.pct_change(20)
# cross-asset breadth as continuous regime multiplier, not common factor removal
breadth=t.gt(0).mean(axis=1); breadth_signal=(breadth-0.5)*2
f=t.mul((1+0.8*breadth_signal),axis=0)
for h in [1,5,10,20]:
 q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(c.shift(-h)/c-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(q,columns=['date','ic','n']).set_index('date'); sd=q.ic.std(ddof=1)
 print('H',h,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/sd,'hit',(q.ic>0).mean())
 for a,b in [('2020','2025-12-31'),('2026','2028-12-31'),('2029','2029-12-31'),('2030','2030-02-07')]:
  x=q[(q.index>=a)&(q.index<=b)]
  if len(x):print(a,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20300207_breadth_conditioned_trend_signal.csv',index=False)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'coverage',out.symbol.nunique()/15)
