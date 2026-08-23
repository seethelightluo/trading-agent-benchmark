import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=3000)
 return d if d is not None and len(d) else get_index_daily_data(s,days=3000)
raw={s:load(s) for s in U}
close=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill()
volu=pd.DataFrame({s:(d.set_index('date').volume if d is not None and 'volume' in d else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().reindex(close.index).ffill()
r=close.pct_change(); ret20=close.pct_change(20); market=ret20.median(axis=1)
# Momentum only when recent participation confirms trend: relative 20d momentum times bounded volume surprise.
vs=(volu.rolling(5).mean()/volu.rolling(60).mean()-1).clip(-1,1)
fac=ret20.sub(market,axis=0)*(1+0.5*vs)
fwd=close.shift(-10)/close-1
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
ic=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(ic),'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean(),fac.rank(axis=1).diff().abs().stack().mean()/14))
for name,x in [('recent180',ic.tail(180)),('recent360',ic.tail(360)),('2030',ic.loc['2030']),('recent60',ic.tail(60))]: print(name,len(x),'IC %.6f ICIR %.6f'%(x.ic.mean(),x.ic.mean()/x.ic.std()))
for h in [1,5,10,20]:
 fw=close.shift(-h)/close-1;q=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('decay',h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q))
fac.to_csv('scripts/miner_3_20310306_volume_confirmed_momentum_signal.csv');ic.to_csv('scripts/miner_3_20310306_volume_confirmed_momentum_ic.csv')
