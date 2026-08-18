import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>300:return x
  except Exception: pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff()
down=r.where(r<0,0.0); dv=down.rolling(30).std()*np.sqrt(252)
rec=r.rolling(5).sum()/(dv+1e-12); trend=r.rolling(40).sum(); f=(rec/(1+trend.abs())).shift(1)
for h in [5,10,20,40]:
 rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z.index=pd.to_datetime(z.index)
 q=z.loc['2026-07-16':'2034-05-25']; print('horizon',h,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
 if h==10:
  for lo,hi in [('2026-07-16','2028-12-31'),('2029','2031-12-31'),('2032','2034-05-25')]:
   a=q.loc[lo:hi]; print('regime',lo,hi,len(a),a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1))
  z.reset_index().to_csv('scripts/miner_1_20340526_downside_efficiency_recovery_ic.csv',index=False)
ranks=f.rank(axis=1,pct=True); print('assets',len(raw),'coverage',f.notna().mean().mean(),'turnover',(ranks-ranks.shift()).abs().mean(axis=1).mean(),'last',f.index.max())
f.to_csv('scripts/miner_1_20340526_downside_efficiency_recovery_signal.csv')
