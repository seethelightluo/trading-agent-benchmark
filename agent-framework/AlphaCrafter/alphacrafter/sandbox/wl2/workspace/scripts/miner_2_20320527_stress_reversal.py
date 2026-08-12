import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=3000)
 if x is not None and len(x):D[s]=x.set_index('date')
P=pd.concat([x.close.rename(s) for s,x in D.items()],axis=1).sort_index().ffill(); r=P.pct_change(); v=r.rolling(20).std()
# Cross-asset stress gate: high median absolute 3d move and broad negative breadth.
r3=r.rolling(3).sum(); disp=r3.std(axis=1); breadth=(r3>0).mean(axis=1)
stress=(disp>=disp.rolling(252,min_periods=60).quantile(.70)) & (breadth<=.35)
f=-(r.rolling(3).sum()/v).where(stress, np.nan).replace([np.inf,-np.inf],np.nan)
def ir(x):return x.mean()/x.std(ddof=1) if len(x)>1 and x.std(ddof=1)>0 else np.nan
rows=[]
for i in range(len(P)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:
  c=z.f.corr(z.y)
  if np.isfinite(c):rows.append((P.index[i],c,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('universe',len(D),'stress_days',int(stress.sum()),'dates',len(q),'avg_n',round(q.n.mean(),3) if len(q) else None,'coverage',round(f.notna().mean().mean(),4),'IC',round(q.ic.mean(),6) if len(q) else None,'ICIR',round(ir(q.ic),6) if len(q) else None,'hit',round((q.ic>0).mean(),4) if len(q) else None)
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 z=q.loc[a:b].ic;print('regime',a,b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(ir(z),6) if len(z)>1 else None)
for h in [1,3,5,10]:
 y=P.pct_change(h).shift(-h); rr=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:rr.append(z.f.corr(z.y))
 print('decay',h,'dates',len(rr),'IC',round(np.nanmean(rr),6) if rr else None)
f.to_csv('scripts/miner_2_20320527_stress_reversal_signal.csv')
