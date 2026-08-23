import numpy as np,pandas as pd, os
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 try:d=get_stock_daily_data(s,days=4200)
 except Exception:d=None
 if d is None or len(d)==0:
  try:d=get_index_daily_data(s,days=4200)
  except Exception:d=None
 return d
raw={s:load(s) for s in U}
px=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill()
try:
 v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']); vix=v.set_index('date')['close'].reindex(px.index).ffill()
except Exception: vix=pd.Series(index=px.index,dtype=float)
mom=px.pct_change(20)
vm=vix.rolling(252,min_periods=126).median(); stress=(vix/vm-1)
# VIX-conditioned trend: trend in calm regime, contrarian reversal in stressed regime
w=((stress-0.15)/0.35).clip(-1,1)
sig=mom*(1-2*w).values[:,None]
sig=pd.DataFrame(sig,index=px.index,columns=px.columns).sub(pd.DataFrame(sig,index=px.index,columns=px.columns).median(axis=1),axis=0)
fwd=px.shift(-10)/px-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
ic=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('range',ic.index.min(),ic.index.max(),'dates',len(ic),'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean(),sig.rank(axis=1,pct=True).diff().abs().stack().mean()))
for name,x in [('recent180',ic.tail(180)),('recent360',ic.tail(360)),('2030',ic.loc['2030']),('recent60',ic.tail(60))]: print(name,len(x),'IC %.6f ICIR %.6f'%(x.ic.mean(),x.ic.mean()/x.ic.std() if x.ic.std() else np.nan))
for h in [1,5,10,20]:
 q=px.shift(-h)/px-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('decay',h,'IC %.6f ICIR %.6f'%(np.nanmean(rr),np.nanmean(rr)/np.nanstd(rr)))
sig.to_csv('scripts/miner_3_20310403_vix_conditioned_trend_signal.csv'); ic.to_csv('scripts/miner_3_20310403_vix_conditioned_trend_ic.csv')
