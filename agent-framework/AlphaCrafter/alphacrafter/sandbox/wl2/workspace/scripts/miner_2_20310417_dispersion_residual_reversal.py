import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); y=r.shift(-1)
shock=r.sub(r.median(axis=1),axis=0)
vol=r.rolling(20,min_periods=10).std()
absdev=shock.sub(shock.median(axis=1),axis=0).abs().median(axis=1)
disp=absdev.rolling(20,min_periods=10).mean(); threshold=disp.rolling(252,min_periods=100).median()
gate=(disp>threshold).astype(float)
f=(-shock/vol).mul(gate,axis=0).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((p.index[i],z.f.corr(z.y),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',round(q.n.mean(),3),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
print('coverage',round(f.notna().mean().mean(),4),'active',round(gate.mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic; print(a,'dates',len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for h in [3,5,10]:
 fy=r.rolling(h).sum().shift(-(h-1)); vals=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),fy.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(z.f.corr(z.y))
 print('decay',h,round(float(np.mean(vals)),6),len(vals))
f.to_csv('scripts/miner_2_20310417_dispersion_residual_reversal_signal.csv')
