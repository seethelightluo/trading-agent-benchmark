import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=4000)
   if d is not None and len(d): return d
  except Exception: pass
S={}
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index(); c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change()
 # Directional efficiency: net 20d move relative to path length, with volatility scaling.
 eff=r.rolling(20).sum()/(r.abs().rolling(20).sum()+1e-12)
 vol=r.rolling(40).std()
 sig=eff/(vol*np.sqrt(20)+1e-12)
 S[s]=pd.DataFrame({'f':sig,'f1':c.shift(-1)/c-1,'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1})
idx=sorted(set().union(*[x.index for x in S.values()]))
for col,h in [('f1',1),('f5',5),('f10',10)]:
 rows=[]
 for dt in idx:
  a=[x.loc[dt,['f',col]].values for x in S.values() if dt in x.index and np.isfinite(x.loc[dt,['f',col]]).all()]
  if len(a)>=8:
   z=pd.DataFrame(a,columns=['f','r']); rows.append((dt,z.f.rank().corr(z.r.rank()),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']);m=q.ic.mean(); ir=m/q.ic.std(ddof=1)*np.sqrt(252)
 print('horizon',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(m,6),'ICIR',round(ir,4),'hit',round((q.ic>0).mean(),4))
 if h==10:
  for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
   w=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic;print('regime',a+'-'+b,len(w),round(w.mean(),6))
