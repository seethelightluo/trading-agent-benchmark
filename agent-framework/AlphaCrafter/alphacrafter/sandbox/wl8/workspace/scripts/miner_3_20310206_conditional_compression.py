import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)==0:d=get_index_daily_data(s,days=3000)
 return d
raw={s:load(s) for s in U}; px=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill(); r=px.pct_change()
# Conditional compression: favor low short/long vol only when cross-asset dispersion is elevated.
disp=r.rolling(20).std().mean(axis=1)
fac=-(r.rolling(10).std()/r.rolling(60).std()).mul((disp/disp.rolling(120).median()),axis=0)
fwd=px.shift(-10)/px-1
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
r0=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('conditional compression dates',len(r0),'avg_n',r0.n.mean(),'coverage',r0.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(r0.ic.mean(),r0.ic.mean()/r0.ic.std(),(r0.ic>0).mean(),fac.rank(axis=1).diff().abs().stack().mean()/14))
for name,x in [('recent180',r0.tail(180)),('recent360',r0.tail(360)),('2030',r0.loc['2030']),('recent60',r0.tail(60))]:print(name,len(x),'%.6f %.6f'%(x.ic.mean(),x.ic.mean()/x.ic.std()))
fac.to_csv('scripts/miner_3_20310206_conditional_compression_signal.csv');r0.to_csv('scripts/miner_3_20310206_conditional_compression_ic.csv')
