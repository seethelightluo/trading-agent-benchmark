import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 return d
px={}
for s in U:
 d=fetch(s)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
D=pd.concat(px,axis=1).sort_index().ffill()
r=D.pct_change()
# Residual momentum: trailing 20d asset return minus trailing 20d equal-weight universe return.
# The common component is removed cross-asset, and the signal is lagged one day.
market=r.mean(axis=1)
sig=(D/D.shift(20)-1).sub((1+market).rolling(20,min_periods=15).apply(np.prod,raw=True)-1,axis=0).shift(1)
# forward returns at requested horizons
rows=[]
for dt in sig.index:
 y=D.shift(-10)/D-1
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,a,b in [('all','2020-01-01','2029-07-15'),('2020-2022','2020-01-01','2022-12-31'),('2023-2024','2023-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027-2028','2027-01-01','2028-12-31'),('recent','2028-09-01','2029-07-15')]:
 q=r.loc[a:b,'ic'].dropna(); print(label,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan,'hit',(q>0).mean() if len(q) else np.nan)
print('dates',len(r),'avg_n',r.n.mean(),'coverage',sig.notna().stack().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().stack().mean())
for h in [1,5,10,20]:
 yy=D.shift(-h)/D-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,'IC',np.nanmean(vals),'dates',len(vals))
out=sig.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_2_20290716_residual_momentum20_signal.csv')
print('artifact scripts/miner_2_20290716_residual_momentum20_signal.csv')
