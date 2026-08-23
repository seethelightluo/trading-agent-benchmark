import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try:
   d=f(s,days=4200)
   if d is not None and len(d): return d
  except Exception: pass
 return None
raw={s:load(s) for s in U}
px=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill()
d=pd.read_csv('../persistent/index_data/DXY.csv'); d['date']=pd.to_datetime(d['date']); macro=d.set_index('date')['close'].reindex(px.index).ffill()
# DXY trend regime: stronger dollar tends to impair risk assets; smoothly tilt cross-sectional momentum toward reversal
mom=px.pct_change(20); dtrend=macro.pct_change(20)
# only lagged information is used: today's signal predicts returns after today
stress=((dtrend-dtrend.rolling(252,min_periods=126).median())/dtrend.rolling(252,min_periods=126).std()).clip(-2,2)
# positive DXY trend reverses momentum, negative trend reinforces it
w=(stress/2).fillna(0)
sig=mom*(1-1.4*w.values[:,None])
sig=pd.DataFrame(sig,index=px.index,columns=px.columns)
sig=sig.sub(sig.median(axis=1),axis=0)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],(px.shift(-10)/px-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
ic=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('range',ic.index.min(),ic.index.max(),'dates',len(ic),'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean(),sig.rank(axis=1,pct=True).diff().abs().stack().mean()))
for name,x in [('recent180',ic.tail(180)),('recent360',ic.tail(360)),('2030',ic.loc['2030']),('recent60',ic.tail(60))]: print(name,len(x),'IC %.6f ICIR %.6f'%(len(x) and x.ic.mean(),x.ic.mean()/x.ic.std() if len(x)>1 and x.ic.std() else np.nan))
for h in [1,5,10,20]:
 rr=[]; q=px.shift(-h)/px-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('decay',h,'IC %.6f ICIR %.6f'%(np.nanmean(rr),np.nanmean(rr)/np.nanstd(rr)))
sig.to_csv('scripts/miner_3_20310417_dxy_conditioned_trend_signal.csv'); ic.to_csv('scripts/miner_3_20310417_dxy_conditioned_trend_ic.csv')
