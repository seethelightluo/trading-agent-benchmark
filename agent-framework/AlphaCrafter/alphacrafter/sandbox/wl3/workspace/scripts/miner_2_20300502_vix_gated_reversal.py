import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None and len(d): D[s]=d.set_index('date')
px=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); lr=np.log(px).diff()
v=get_index_daily_data('VIX',4000)
if v is None: raise RuntimeError('VIX unavailable')
v=v.set_index('date').close.astype(float).reindex(px.index).ffill()
# High-volatility conditional short-term reversal. VIX state is lagged, and signal is inactive in ordinary regimes.
r=lr.rolling(3).sum(); vol=lr.rolling(20).std()*np.sqrt(3)+1e-9
shock=(-r/vol).clip(-4,4)
state=(v > v.rolling(60).median()).astype(float)
f=(shock*state).shift(1)
f=f.sub(f.median(axis=1),axis=0)
for h in [1,3,5,10]:
 rows=[]
 for i,dt in enumerate(px.index[:-h]):
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
 print('H',h,'obs',len(q),'avgN %.2f'%q.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  y=q.loc[a:b].ic
  if len(y): print(a+'-'+b,len(y),'%.6f'%y.mean(),'%.6f'%(y.mean()/y.std()))
rr=f.rank(axis=1,pct=True)
print('dates',len(px),'instruments',len(D),'coverage %.4f'%(f.notna().sum().sum()/(len(f)*len(D))),'turnover %.4f'%(rr.diff().abs().mean(axis=1).mean()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300502_vix_gated_reversal_signal.csv',index=False)
