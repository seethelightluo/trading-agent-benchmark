import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:x=get_index_daily_data(s,4000)
 except:x=None
 if x is None or len(x)<300:
  try:x=get_stock_daily_data(s,4000)
  except:x=None
 if x is not None and len(x): D[s]=x.set_index(pd.to_datetime(x.date)).close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); ds=r.where(r<0,0).rolling(20,min_periods=15).std().shift(1)
# Contrarian response to recent downside shocks, normalized by downside risk
f=(-(p.shift(1).pct_change(10)))/(ds+1e-8); f=f.replace([np.inf,-np.inf],np.nan); out=[]
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fr.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8:q.append(z.f.corr(z.r))
 q=pd.Series(q).dropna();print('h',h,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 if h==10:
  for n in [260,520,1040,1560]:
   a=q.tail(n); print('recent',n,'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('assets',len(D),'dates',len(p),'coverage %.4f'%(f.notna().sum(axis=1).mean()/15),'turnover %.4f'%(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
