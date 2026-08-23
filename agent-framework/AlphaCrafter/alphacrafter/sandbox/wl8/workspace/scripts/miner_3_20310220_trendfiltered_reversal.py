import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)==0:d=get_index_daily_data(s,days=3000)
 return d
raw={s:load(s) for s in U};px=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill()
# Robust short-horizon reversal: negative 3-day return, cross-section demeaned,
# with a mild 20-day trend interaction to avoid buying persistent breakdowns.
r3=px.pct_change(3); trend=px.pct_change(20)
fac=-r3*(1+np.maximum(-trend,0))
fwd=px.shift(-10)/px-1; rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
ic=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('trendfiltered 3d reversal dates',len(ic),'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean(),fac.rank(axis=1).diff().abs().stack().mean()/14))
for name,x in [('recent180',ic.tail(180)),('recent360',ic.tail(360)),('2030',ic.loc['2030']),('recent60',ic.tail(60))]:print(name,len(x),'%.6f %.6f'%(x.ic.mean(),x.ic.mean()/x.ic.std() if x.ic.std()>0 else np.nan))
fac.to_csv('scripts/miner_3_20310220_trendfiltered_reversal_signal.csv');ic.to_csv('scripts/miner_3_20310220_trendfiltered_reversal_ic.csv')
