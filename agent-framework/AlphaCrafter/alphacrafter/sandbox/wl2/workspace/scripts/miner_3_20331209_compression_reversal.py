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
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff(); v20=r.rolling(20).std(); v60=r.rolling(60).std()
# compression and medium-term reversal, lagged to avoid look-ahead
f=((-v20/v60).rank(axis=1,pct=True) - r.rolling(60).sum().rank(axis=1,pct=True))/2
f=f.shift(1)
for h in [10,20]:
 fr=p.shift(-h)/p-1; rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=z.ic.mean(); print('H',h,'dates',len(z),'avgN',z.n.mean(),'IC',ic,'ICIR',ic/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
 for lo,hi in [('2026','2029-12-31'),('2030','2033-12-31')]:
  q=z.loc[lo:hi]; print('REG',lo,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
 if h==20:z.to_csv('scripts/miner_3_20331209_compression_reversal_20d_ic.csv')
print('assets',len(raw),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
