import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try:
   x=f(s,days=4200)
   if x is not None and len(x): return x
  except: pass
raw={s:load(s) for s in U}; px=pd.DataFrame({s:(x.set_index('date').close if x is not None else pd.Series(dtype=float)) for s,x in raw.items()}).sort_index().ffill()
a=pd.read_csv('../persistent/index_data/DXY.csv'); a.date=pd.to_datetime(a.date); dx=a.set_index('date').close.reindex(px.index).ffill()
# DXY shock-conditioned short-horizon reversal; all inputs are lagged close observations
r5=px.pct_change(5); shock=dx.pct_change(5); z=(shock-shock.rolling(252,min_periods=126).mean())/shock.rolling(252,min_periods=126).std()
# stress amplifies reversal; calm periods retain a small reversal component
sig=-r5*(1+0.8*z.clip(-1,2).fillna(0).values[:,None]); sig=pd.DataFrame(sig,index=px.index,columns=px.columns); sig=sig.sub(sig.median(axis=1),axis=0)
fwd=px.shift(-10)/px-1; rows=[]
for dt in sig.index:
 q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
ic=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('range',ic.index.min(),ic.index.max(),'dates',len(ic),'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean(),sig.rank(axis=1,pct=True).diff().abs().stack().mean()))
for name,x in [('recent180',ic.tail(180)),('recent360',ic.tail(360)),('2030',ic.loc['2030']),('recent60',ic.tail(60))]: print(name,len(x),'IC %.6f ICIR %.6f'%(x.ic.mean(),x.ic.mean()/x.ic.std() if len(x)>1 and x.ic.std() else np.nan))
for h in [1,5,10,20]:
 q=[]; f=px.shift(-h)/px-1
 for dt in sig.index:
  x=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(x)>=8:q.append(x.iloc[:,0].rank().corr(x.iloc[:,1].rank()))
 print('decay',h,'IC %.6f ICIR %.6f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q)))
sig.to_csv('scripts/miner_3_20310417_dxy_shock_reversal_signal.csv');ic.to_csv('scripts/miner_3_20310417_dxy_shock_reversal_ic.csv')
