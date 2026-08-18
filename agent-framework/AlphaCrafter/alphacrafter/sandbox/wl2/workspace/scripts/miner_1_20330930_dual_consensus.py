import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d)>150:return d
  except Exception: pass
xs={s:fetch(s) for s in U}; xs={s:d for s,d in xs.items() if d is not None}
# Candidate: lagged dual-horizon trend consensus. Cross-sectional rank-like raw
# 20d trend is weighted by agreement with 60d trend, then scaled by 20d vol.
def panel(h):
 a=[]
 for s,d in xs.items():
  c=d.close.astype(float); lr=np.log(c/c.shift(1)); vol=lr.rolling(20).std()
  t20=np.log(c/c.shift(20)); t60=np.log(c/c.shift(60))
  agree=np.sign(t20)*np.sign(t60)
  f=(t20/vol)*agree.clip(lower=0).shift(1)
  r=c.shift(-h)/c-1
  a.append(pd.DataFrame({'date':d.date,'f':f,'r':r,'s':s}).dropna())
 x=pd.concat(a); rows=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: rows.append((dt,g.f.corr(g.r,method='spearman'),len(g)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [1,3,5,10]:
 q=panel(h); print('h=%d dates=%d avg_n=%.3f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(q),q.n.mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
q=panel(1)
for a,b in [('2020','2025'),('2026','2029'),('2030','2033')]:
 z=q.loc[a:b]; print('regime=%s-%s dates=%d IC=%.6f ICIR=%.6f'%(a,b,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)))
print('assets=%d'%len(xs))
