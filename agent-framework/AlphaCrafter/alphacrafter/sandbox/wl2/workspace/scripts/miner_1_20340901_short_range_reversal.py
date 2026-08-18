import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>300:return x
  except: pass
raw={s:fetch(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff()
# Candidate: short-horizon reversal damped by recent volatility, with 60d range location.
# Low range-location assets that recently fell are preferred; volatility scaling avoids risk concentration.
ret5=np.log(p/p.shift(5)); vol20=r.rolling(20).std(); loc60=(p-p.rolling(60).min())/(p.rolling(60).max()-p.rolling(60).min()+1e-12)
f=(-ret5/(vol20+1e-12))*(1-loc60)
f=f.shift(1)
def calc(h):
 rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.loc['2026-07-16':'2034-08-30']
 return q
for h in [1,3,5,10,20]:
 q=calc(h); sd=q.ic.std(ddof=1); print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/sd,6),'hit',round((q.ic>0).mean(),4))
for a,b in [('2026-07-16','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2034-08-30')]:
 q=calc(10).loc[a:b]; print('REG',a,b,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6))
q=calc(10); print('coverage',f.loc[q.index].notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_1_20340901_short_range_reversal_signal.csv')
calc(10).reset_index().to_csv('scripts/miner_1_20340901_short_range_reversal_ic.csv',index=False)
