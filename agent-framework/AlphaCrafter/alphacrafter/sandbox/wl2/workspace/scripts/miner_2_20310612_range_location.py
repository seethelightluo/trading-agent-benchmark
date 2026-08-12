import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None: D[s]=x.set_index('date')
cl=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index().ffill(); r=cl.pct_change(); y=r.shift(-1)
# Short-horizon range-location reversal: assets closing near the top/bottom of their 10-day range tend to mean-revert next day.
lo=cl.rolling(10,min_periods=8).min(); hi=cl.rolling(10,min_periods=8).max(); f=0.5-(cl-lo)/(hi-lo+1e-12)
rows=[]
for i in range(len(f)-1):
 z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((f.index[i],z.f.corr(z.y),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
print('universe',len(D),'dates',len(q),'avgN',round(q.n.mean(),3),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,(q.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic; print(a+'-'+b,'dates',len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('coverage %.4f turnover %.4f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
f.to_csv('scripts/miner_2_20310612_range_location_signal.csv')
