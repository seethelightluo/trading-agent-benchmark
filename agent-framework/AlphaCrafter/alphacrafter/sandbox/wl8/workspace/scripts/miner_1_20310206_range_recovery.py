import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)==0:d=get_index_daily_data(s,days=3000)
 return d
raw={s:load(s) for s in U}; px=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill(); r=px.pct_change()
peak=px.rolling(60).max(); draw=px/peak-1
fac=r.rolling(10).sum()*(-draw).where(draw<=-.05,0)
fwd=px.shift(-10)/px-1
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
r0=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(r0),'avg_n',r0.n.mean(),'coverage',r0.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(r0.ic.mean(),r0.ic.mean()/r0.ic.std(),(r0.ic>0).mean()))
for name,x in [('recent60',r0.tail(60)),('recent120',r0.tail(120)),('recent252',r0.tail(252)),('2030',r0.loc['2030'])]: print(name,len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().stack().mean())
fac.to_csv('scripts/miner_1_20310206_range_recovery_signal.csv');r0.to_csv('scripts/miner_1_20310206_range_recovery_ic.csv')
