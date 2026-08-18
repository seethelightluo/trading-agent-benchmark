import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=4500)
 if d is None or len(d)<200: d=get_index_daily_data(s,days=4500)
 if d is not None: D[s]=d.set_index('date')['close'].astype(float)
C=pd.concat(D,axis=1).sort_index().ffill(); R=C.pct_change()
# Dispersion expansion: use change in 20d cross-sectional volatility versus its 60d level.
# In expansion regimes, relative 5d leadership is expected to persist; scale by idiosyncratic vol.
csdisp=R.std(axis=1); expansion=(csdisp/csdisp.rolling(60,min_periods=40).mean()-1)
active=(expansion>0.10) & (expansion>expansion.rolling(252,min_periods=126).median())
ret5=C/C.shift(5)-1; vol20=R.rolling(20).std()
f=(ret5.sub(ret5.mean(axis=1),axis=0)/vol20.replace(0,np.nan)).mul(active.astype(float),axis=0)
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(C.shift(-10)/C-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
ic=pd.Series({d:v for d,v,n in rows})
print('dates',len(ic),'avg_n',np.mean([n for d,v,n in rows]),'coverage',len(ic)/len(f))
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for h in [3,5,10,20]:
 q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(C.shift(-h)/C-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q).dropna();print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'n',len(q))
for name,sel in [('2020-25',ic.index<'2026-01-01'),('2026-29',(ic.index>='2026-01-01')&(ic.index<'2030-01-01')),('2030+',ic.index>='2030-01-01'),('last365',ic.index>=ic.index.max()-pd.Timedelta(days=365))]:
 q=ic[sel]; print(name,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>2 else np.nan,(q>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'valid_assets',C.notna().mean().mean(),'active_rate',active.mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20330916_dispersion_expansion_momentum_signal.csv',index=False)
