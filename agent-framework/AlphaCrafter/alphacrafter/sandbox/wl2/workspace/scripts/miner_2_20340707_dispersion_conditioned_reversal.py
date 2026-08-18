import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None:return x
  except Exception: pass
raw={s:get(s) for s in U}; p=pd.concat({s:x.set_index('date').close for s,x in raw.items() if x is not None},axis=1).sort_index(); r=np.log(p).diff()
r5=r.rolling(5,min_periods=5).sum(); cross_disp=r.rolling(20,min_periods=10).sum().std(axis=1)
mult=(cross_disp/cross_disp.rolling(120,min_periods=30).median()).clip(.5,2.0)
f=(-r5).mul(mult,axis=0).rank(axis=1,pct=True).shift(1)
for h in [1,5,10,20,40]:
 o=[]
 for d in f.index:
  a=pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(a)>=8:o.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 q=pd.DataFrame(o,columns=['d','ic','n']).set_index('d').loc['2026-07-16':'2034-07-05']
 print('H',h,'dates',len(q),'N',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 for a,b in [('2026-07-16','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2034-07-05')]:
  z=q.loc[a:b];print('REG',a,len(z),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1),6))
print('coverage',round(f.loc['2026-07-16':'2034-07-05'].notna().mean().mean(),5),'turnover',round(f.diff().abs().mean().mean(),5))
f.rename_axis('date').to_csv('scripts/miner_2_20340707_dispersion_conditioned_reversal_signal.csv')
