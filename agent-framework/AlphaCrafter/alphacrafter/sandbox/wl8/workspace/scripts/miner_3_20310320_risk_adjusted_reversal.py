import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)==0: d=get_index_daily_data(s,days=3000)
 return d
raw={s:load(s) for s in U}
px=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill()
# Candidate: risk-adjusted short-horizon reversal. Normalize the recent 5d shock
# by trailing 20d realized volatility, then demean cross-sectionally.
r5=px.pct_change(5); vol=px.pct_change().rolling(20).std()*np.sqrt(20)
fac=(-r5/vol).sub((-r5/vol).median(axis=1),axis=0)
fwd=px.shift(-10)/px-1
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
ic=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(ic),'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean(),fac.rank(axis=1).diff().abs().stack().mean()/14))
for name,x in [('recent180',ic.tail(180)),('recent360',ic.tail(360)),('2030',ic.loc['2030']),('recent60',ic.tail(60))]: print(name,len(x),'IC %.6f ICIR %.6f'%(x.ic.mean(),x.ic.mean()/x.ic.std() if x.ic.std() else np.nan))
# decay at several forward horizons
for h in [1,5,10,20]:
 q=px.shift(-h)/px-1; rr=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('decay',h,'IC %.6f ICIR %.6f'%(np.nanmean(rr),np.nanmean(rr)/np.nanstd(rr)))
fac.to_csv('scripts/miner_3_20310320_risk_adjusted_reversal_signal.csv'); ic.to_csv('scripts/miner_3_20310320_risk_adjusted_reversal_ic.csv')
