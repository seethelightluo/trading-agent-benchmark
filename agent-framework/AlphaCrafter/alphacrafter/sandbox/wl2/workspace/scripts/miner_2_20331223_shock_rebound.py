import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>200:return x
  except Exception: pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff()
# Broad-shock residual rebound: reverse 3d return only when cross-sectional
# median absolute 3d move is in its trailing 60d upper quartile.
ret3=r.rolling(3).sum(); shock=ret3.abs().median(axis=1); q=shock.rolling(60,min_periods=40).quantile(.75)
active=(shock>q).astype(float); f=(-ret3.mul(active,axis=0)).rank(axis=1,pct=True).shift(1)
print('candidate shock_rebound assets',len(raw),'dates',len(p),'active',round(active.mean(),4))
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=z.ic.mean(); ir=ic/z.ic.std(ddof=1)
 print('H',h,'dates',len(z),'avgN',round(z.n.mean(),3),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((z.ic>0).mean(),4))
 for lo,hi in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-12-31')]:
  zz=z.loc[lo:hi]
  if len(zz): print(' REG',lo,len(zz),round(zz.ic.mean(),6),round(zz.ic.mean()/zz.ic.std(ddof=1),6))
 if h==10:z.to_csv('scripts/miner_2_20331223_shock_rebound_ic.csv')
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.diff().abs().mean(axis=1).mean(),6))
