import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   z=fn(s,days=6000)
   if z is not None:return z
  except: pass
raw={s:fch(s) for s in U}; p=pd.concat({s:z.set_index('date').close for s,z in raw.items() if z is not None},axis=1).sort_index(); r=np.log(p).diff()
# low-volatility quality plus short-term reversal, lagged one day
v20=r.rolling(20,min_periods=10).std(); v60=r.rolling(60,min_periods=30).std()
f=( -v20.rank(axis=1,pct=True) + 0.55*(-r.rolling(5,min_periods=5).sum()).rank(axis=1,pct=True) + 0.25*(-v60).rank(axis=1,pct=True) ).rank(axis=1,pct=True).shift(1)
for h in [1,5,10,20]:
 o=[]
 for d in f.index:
  a=pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(a)>=8:o.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 q=pd.DataFrame(o,columns=['d','ic','n']).set_index('d').loc['2026-07-16':'2034-06-22']
 print('H',h,'dates',len(q),'N',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 for a,b in [('2026-07-16','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2034-06-22')]:
  z=q.loc[a:b];print(a,len(z),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1),6))
print('coverage_valid',f.loc['2026-07-16':'2034-06-22'].notna().mean().mean(),'turn',f.diff().abs().mean().mean())
f.rename_axis('date').to_csv('../persistent/miner_3_20340623_lowvol_reversal_signal.csv')
