import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=3000)
 if x is not None and len(x):D[s]=x.set_index('date')
P={k:pd.concat({s:x[k] for s,x in D.items()},axis=1).sort_index().ffill() for k in ['close','high','low']}
r=P['close'].pct_change(); rng=(P['high']-P['low']).replace(0,np.nan); clv=2*(P['close']-P['low'])/rng-1
v=r.rolling(20).std(); f=-(r/v)*(-clv); f=f.replace([np.inf,-np.inf],np.nan)
def ir(x):return x.mean()/x.std(ddof=1) if len(x)>1 and x.std(ddof=1)>0 else np.nan
rows=[]
for i in range(len(r)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:
  c=z.f.corr(z.y)
  if np.isfinite(c):rows.append((r.index[i],c,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('universe',len(D),'dates',len(q),'avg_n',q.n.mean(),'IC',q.ic.mean(),'ICIR',ir(q.ic),'hit',(q.ic>0).mean(),'coverage',f.notna().mean().mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 z=q.loc[a:b].ic;print('regime',a,b,len(z),z.mean() if len(z) else None,ir(z) if len(z)>1 else None)
for h in [1,3,5,10]:
 y=P['close'].pct_change(h).shift(-h); rr=[]
 for i in range(len(r)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:rr.append(z.f.corr(z.y))
 print('decay',h,len(rr),np.nanmean(rr))
f.to_csv('scripts/miner_2_20320513_clv_1d_signal.csv')
