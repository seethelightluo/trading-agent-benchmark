import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None:return d
  except Exception: pass
S={}
for s in SYMS:
 d=fetch(s)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); c=d.set_index('date').sort_index().close; r=c.pct_change()
 cons=r.gt(0).rolling(20).mean()-r.lt(0).rolling(20).mean()
 sig=cons*(c.pct_change(20)/(r.rolling(20).std()*np.sqrt(20)))
 S[s]=pd.DataFrame({'sig':sig.shift(1),**{f'f{h}':c.shift(-h)/c-1 for h in [5,10,20]}})
print('assets',len(S),'dates',len(set().union(*[x.index for x in S.values()])))
for h in [5,10,20]:
 out=[]
 for dt in sorted(set().union(*[x.index for x in S.values()])):
  a=[(x.loc[dt].sig,x.loc[dt][f'f{h}']) for x in S.values() if dt in x.index and np.isfinite(x.loc[dt].sig) and np.isfinite(x.loc[dt][f'f{h}'])]
  if len(a)>=8:
   z=pd.DataFrame(a,columns=['s','r']); out.append((dt,z.s.rank().corr(z.r.rank()),len(a)))
 q=pd.DataFrame(out,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,4),'IC',round(m,7),'daily_ICIR',round(m/sd,7),'hit',round((q.ic>0).mean(),4),'turnover',round(pd.DataFrame({k:v.sig.rank(pct=True) for k,v in S.items()}).diff().abs().mean().mean(),5))
 print('regimes',q.assign(y=q.date.dt.year).groupby('y').ic.mean().tail(8).round(5).to_dict())
